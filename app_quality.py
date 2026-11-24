import glob
import os
import sys
import time
import json
import gradio as gr
import numpy as np
import openvino as ov

# Определяем папку, где лежит .exe
if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)

core = ov.Core()

# --- ФУНКЦИИ ДЛЯ РАБОТЫ С НАСТРОЙКАМИ (JSON) ---
def get_config():
    config_file = os.path.join(APP_DIR, "settings.json")
    default_config = {
        "limit_tile_256_models": [
            "4xNomos8kDAT.onnx"
        ]
    }
    
    if not os.path.exists(config_file):
        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Не удалось создать settings.json: {e}")
        return default_config
        
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка чтения settings.json: {e}")
        return default_config

# Новая функция для сохранения настроек прямо из интерфейса
def save_config_ui(models_string):
    # Разбиваем строку по запятым, убираем лишние пробелы
    models_list = [m.strip() for m in models_string.split(",") if m.strip()]
    config_data = {"limit_tile_256_models": models_list}
    config_file = os.path.join(APP_DIR, "settings.json")
    
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        return "✅ Настройки успешно сохранены в settings.json!"
    except Exception as e:
        return f"❌ Ошибка при сохранении: {e}"

# -----------------------------------------------

def get_best_device():
    devices = core.available_devices
    best_gpu = "CPU"
    for d in devices:
        if "GPU" in d:
            try:
                name = core.get_property(d, "FULL_DEVICE_NAME")
                if "Arc" in name or "Intel" in name:
                    return d
                best_gpu = d
            except:
                best_gpu = d
    return best_gpu

DEVICE = get_best_device()
compiled_cache = {}


def get_available_models():
    models = sorted(glob.glob("*.onnx"))
    return models if models else ["Нет .onnx файлов в папке с программой!"]


def get_compiled_model(model_name):
    if model_name not in compiled_cache:
        model = core.read_model(model_name)
        
        config_ov = {}
        if DEVICE != "CPU":
            if "dat" in model_name.lower():
                config_ov = {"INFERENCE_PRECISION_HINT": "f32"}
            else:
                config_ov = {"INFERENCE_PRECISION_HINT": "f16"}
                
        compiled = core.compile_model(model, device_name=DEVICE, config=config_ov)
        compiled_cache[model_name] = compiled
    return compiled_cache[model_name]


def make_blend_mask(h, w, overlap):
    mask_y = np.ones(h, dtype=np.float32)
    mask_x = np.ones(w, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0.05, 1.0, overlap, dtype=np.float32)
        mask_y[:overlap] *= ramp
        mask_y[-overlap:] *= ramp[::-1]
        mask_x[:overlap] *= ramp
        mask_x[-overlap:] *= ramp[::-1]
    return (mask_y[:, None] * mask_x[None, :])[:, :, None]


def run_single_pass(img_rgb, compiled, tile_size=512, overlap=32, scale=4):
    h, w, _ = img_rgb.shape
    out_h, out_w = h * scale, w * scale
    output = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weights = np.zeros((out_h, out_w, 1), dtype=np.float32)

    step = tile_size - overlap
    jobs = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            y1 = max(0, y2 - tile_size)
            x1 = max(0, x2 - tile_size)

            tile = np.ascontiguousarray(img_rgb[y1:y2, x1:x2], dtype=np.float32) / 255.0
            tensor = np.ascontiguousarray(np.transpose(tile, (2, 0, 1))[None, ...])
            jobs.append((tensor, y1, y2, x1, x2))

    infer_queue = ov.AsyncInferQueue(compiled, 4)
    results = [None] * len(jobs)

    def callback(request, userdata):
        results[userdata] = request.get_output_tensor(0).data[0].copy()

    infer_queue.set_callback(callback)

    for i, (tensor, _, _, _, _) in enumerate(jobs):
        infer_queue.start_async({0: tensor}, userdata=i)

    infer_queue.wait_all()

    for i, (_, y1, y2, x1, x2) in enumerate(jobs):
        pred = results[i]
        pred = np.clip(np.transpose(pred, (1, 2, 0)), 0.0, 1.0)

        oy1, oy2 = y1 * scale, y2 * scale
        ox1, ox2 = x1 * scale, x2 * scale
        mask = make_blend_mask(oy2 - oy1, ox2 - ox1, overlap * scale)

        output[oy1:oy2, ox1:ox2] += pred * mask
        weights[oy1:oy2, ox1:ox2] += mask

    return output / np.maximum(weights, 1e-6)


def upscale_hq(img_rgb, model_name, quality_mode, tile_size):
    if img_rgb is None or not os.path.exists(model_name):
        return None, "Ошибка: загрузите картинку и проверьте наличие .onnx файла"

    app_config = get_config()
    restricted_models = app_config.get("limit_tile_256_models", [])
    
    if model_name in restricted_models and tile_size > 256:
        error_msg = f"⚠️ ОШИБКА: Для модели '{model_name}' установлен лимит тайла 256 во избежание сбоев.\nСнизьте размер тайла или измените настройки программы."
        return None, error_msg

    t0 = time.perf_counter()
    compiled = get_compiled_model(model_name)

    if "x8" in quality_mode:
        transforms = [(k, flip) for flip in (False, True) for k in range(4)]
    elif "x4" in quality_mode:
        transforms = [(k, False) for k in range(4)]
    else:
        transforms = [(0, False)]

    acc = None
    for k, flip in transforms:
        aug = np.rot90(img_rgb, k)
        if flip:
            aug = np.fliplr(aug)

        pred = run_single_pass(aug, compiled, tile_size=tile_size)

        if flip:
            pred = np.fliplr(pred)
        pred = np.rot90(pred, -k)

        acc = pred if acc is None else (acc + pred)

    result = (np.clip(acc / len(transforms), 0.0, 1.0) * 255.0).astype(np.uint8)
    elapsed = time.perf_counter() - t0
    
    try:
        real_device_name = core.get_property(DEVICE, "FULL_DEVICE_NAME")
    except:
        real_device_name = DEVICE

    out_h, out_w, _ = result.shape
    info = f"Готово за {elapsed:.2f} сек (На: {real_device_name}) | {model_name} | {out_w}×{out_h}"
    return result, info


models_list = get_available_models()
# Получаем текущие настройки для отображения в интерфейсе при запуске
initial_config = get_config()
initial_models_str = ", ".join(initial_config.get("limit_tile_256_models", []))

with gr.Blocks(title="AI Upscaler Pro") as demo:
    gr.Markdown(f"## AI Upscaler Pro (Движок: OpenVINO)")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Исходное изображение")
            model_dropdown = gr.Dropdown(
                choices=models_list,
                value=models_list[0] if models_list else None,
                label="Нейросеть",
            )
            quality_radio = gr.Radio(
                choices=[
                    "Быстрый (1 прогон)",
                    "Супер-качество (Self-Ensemble x4)",
                    "Ультра-максимум (Self-Ensemble x8)",
                ],
                value="Супер-качество (Self-Ensemble x4)",
                label="Режим обработки",
            )
            tile_slider = gr.Slider(256, 768, value=512, step=128, label="Размер тайла")
            with gr.Row():
                clear_btn = gr.ClearButton([input_img], value="Очистить")
                run_btn = gr.Button("Улучшить качество", variant="primary")

        with gr.Column():
            output_img = gr.Image(type="numpy", format="png", label="Результат")
            stats_box = gr.Textbox(label="Статистика / Ошибки", interactive=False)

    # --- НОВЫЙ БЛОК НАСТРОЕК В ИНТЕРФЕЙСЕ ---
    with gr.Accordion("⚙️ Настройки ограничений (settings.json)", open=False):
        gr.Markdown("Укажите через запятую названия моделей, для которых размер тайла должен быть строго ограничен до 256. Оставьте поле пустым, чтобы снять лимиты со всех моделей.")
        with gr.Row():
            settings_input = gr.Textbox(
                label="Модели с лимитом (через запятую)", 
                value=initial_models_str,
                placeholder="Например: 4xNomos8kDAT.onnx, другая_модель.onnx",
                scale=4
            )
            save_settings_btn = gr.Button("Сохранить", scale=1)
        
        save_status = gr.Textbox(show_label=False, interactive=False)

    # Обработчики кнопок
    run_btn.click(
        fn=upscale_hq,
        inputs=[input_img, model_dropdown, quality_radio, tile_slider],
        outputs=[output_img, stats_box],
    )
    clear_btn.add([output_img, stats_box])
    
    save_settings_btn.click(
        fn=save_config_ui,
        inputs=[settings_input],
        outputs=[save_status]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
 