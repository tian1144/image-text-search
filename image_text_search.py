# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import shutil
import json
import pytesseract
from PIL import Image, ImageEnhance
import threading
import hashlib
import re
import concurrent.futures
import subprocess
from pathlib import Path

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

try:
    import paddle
except Exception:
    paddle = None

def resolve_tesseract_path():
    """优先使用常见安装路径，其次尝试从环境变量中查找"""
    common_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(common_path):
        return common_path

    path_from_env = shutil.which('tesseract')
    if path_from_env:
        return path_from_env

    return common_path

# 设置Tesseract路径（支持自动探测）
pytesseract.pytesseract.tesseract_cmd = resolve_tesseract_path()

class ImageTextSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图片文本搜索工具")
        self.root.geometry("640x520")
        self.root.resizable(False, False)
        
        # 创建主框架
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 文本输入区域
        ttk.Label(main_frame, text="请输入要查找的文本:", font=('微软雅黑', 10)).pack(anchor=tk.W, pady=(0, 5))
        self.text_var = tk.StringVar()
        text_entry = ttk.Entry(main_frame, textvariable=self.text_var, font=('微软雅黑', 12), width=40)
        text_entry.pack(anchor=tk.W, pady=(0, 20))
        
        # 文件夹选择按钮
        ttk.Button(main_frame, text="选择文件夹", command=self.select_folder, style='TButton').pack(anchor=tk.W, pady=(0, 20))
        self.folder_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.folder_var, font=('微软雅黑', 9), foreground='gray').pack(anchor=tk.W, pady=(0, 20))

        # 识别模式选择（速度/准确率平衡）
        mode_frame = ttk.Frame(main_frame)
        mode_frame.pack(anchor=tk.W, pady=(0, 15))
        ttk.Label(mode_frame, text="OCR引擎:", font=('微软雅黑', 10)).pack(side=tk.LEFT)
        self.ocr_engine_var = tk.StringVar(value="Tesseract(兼容)")
        ttk.Combobox(
            mode_frame,
            textvariable=self.ocr_engine_var,
            values=["Tesseract(兼容)", "PaddleOCR-GPU(高性能)"],
            state="readonly",
            width=20
        ).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Label(mode_frame, text="识别模式:", font=('微软雅黑', 10)).pack(side=tk.LEFT)
        self.ocr_mode_var = tk.StringVar(value="极速")
        ttk.Combobox(
            mode_frame,
            textvariable=self.ocr_mode_var,
            values=["极速", "平衡", "精准"],
            state="readonly",
            width=8
        ).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Label(mode_frame, text="并行线程:", font=('微软雅黑', 10)).pack(side=tk.LEFT)
        self.worker_var = tk.StringVar(value="")
        ttk.Entry(mode_frame, textvariable=self.worker_var, width=6).pack(side=tk.LEFT, padx=(8, 0))

        # GPU/核显加速开关（实验性）
        self.gpu_checkbox_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            main_frame,
            text="启用GPU/核显加速（实验）",
            variable=self.gpu_checkbox_var
        ).pack(anchor=tk.W, pady=(0, 12))

        # 缓存管理区域
        cache_frame = ttk.Frame(main_frame)
        cache_frame.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(cache_frame, text="缓存路径:", font=('微软雅黑', 10)).pack(anchor=tk.W)
        self.cache_path_var = tk.StringVar(value="")
        ttk.Label(cache_frame, textvariable=self.cache_path_var, font=('微软雅黑', 9), foreground='gray').pack(anchor=tk.W, pady=(2, 6))
        cache_btn_frame = ttk.Frame(cache_frame)
        cache_btn_frame.pack(anchor=tk.W)
        ttk.Button(cache_btn_frame, text="删除缓存", command=self.clear_cache).pack(side=tk.LEFT)
        ttk.Button(cache_btn_frame, text="切换缓存路径", command=self.change_cache_path).pack(side=tk.LEFT, padx=(10, 0))
        
        # 开始搜索按钮
        self.search_button = ttk.Button(main_frame, text="开始搜索", command=self.start_search, style='TButton')
        self.search_button.pack(anchor=tk.W, pady=(0, 10))
        
        # 状态标签
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(main_frame, textvariable=self.status_var, font=('微软雅黑', 9)).pack(anchor=tk.W)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(10, 0))
        
        # 初始化变量
        self.target_folder = ""
        self.image_index = {}
        self.index_version = 2
        self.gpu_accel_enabled = False
        self.gpu_backend = "CPU"
        self.gpu_path_used_count = 0
        self._thread_local = threading.local()
        self._paddle_error_logged = False
        self.default_index_file = os.path.join(os.path.expanduser("~"), ".image_text_search_index.json")
        self.config_file = os.path.join(os.path.expanduser("~"), ".image_text_search_config.json")
        self.index_file = self.default_index_file
        self.load_config()
        self.load_index()
        self.refresh_cache_path_label()
        
    def select_folder(self):
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if folder:
            self.target_folder = folder
            self.folder_var.set(folder)
    
    def load_index(self):
        """加载图片文本索引"""
        try:
            if os.path.exists(self.index_file):
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "items" in data:
                        self.image_index = data.get("items", {})
                    elif isinstance(data, dict):
                        self.image_index = data
                    else:
                        self.image_index = {}
        except Exception as e:
            print(f"加载索引出错: {e}")
            self.image_index = {}

    def load_config(self):
        """加载配置（缓存路径等）"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                configured_path = data.get("index_file")
                if configured_path:
                    self.index_file = configured_path
        except Exception as e:
            print(f"加载配置出错: {e}")
            self.index_file = self.default_index_file

    def save_config(self):
        """保存配置（缓存路径等）"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump({"index_file": self.index_file}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置出错: {e}")

    def refresh_cache_path_label(self):
        self.cache_path_var.set(self.index_file)
    
    def save_index(self):
        """保存图片文本索引"""
        try:
            parent = os.path.dirname(self.index_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {"version": self.index_version, "items": self.image_index},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            print(f"保存索引出错: {e}")

    def clear_cache(self):
        """清空缓存索引"""
        confirm = messagebox.askyesno("确认", "确定要删除当前缓存吗？此操作不可恢复。")
        if not confirm:
            return
        try:
            self.image_index = {}
            if os.path.exists(self.index_file):
                os.remove(self.index_file)
            messagebox.showinfo("完成", "缓存已删除。")
        except Exception as e:
            messagebox.showerror("错误", f"删除缓存失败: {e}")

    def change_cache_path(self):
        """切换缓存文件保存路径，并迁移历史缓存"""
        target_dir = filedialog.askdirectory(title="选择新的缓存保存文件夹")
        if not target_dir:
            return
        try:
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            new_index_file = os.path.join(target_dir, "image_text_search_index.json")
            old_index_file = self.index_file

            if os.path.abspath(new_index_file) != os.path.abspath(old_index_file):
                if os.path.exists(old_index_file):
                    if os.path.exists(new_index_file):
                        os.remove(new_index_file)
                    shutil.move(old_index_file, new_index_file)

                self.index_file = new_index_file
                self.save_config()
                self.load_index()
                if self.image_index:
                    self.save_index()
                self.refresh_cache_path_label()
                messagebox.showinfo("完成", "缓存路径已切换，旧缓存已迁移到新路径。")
            else:
                messagebox.showinfo("提示", "新路径与当前缓存路径一致，无需切换。")
        except Exception as e:
            messagebox.showerror("错误", f"切换缓存路径失败: {e}")
    
    def get_image_hash(self, image_path):
        """生成图片的唯一哈希值"""
        try:
            stat = os.stat(image_path)
            hash_input = f"{image_path}_{stat.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception as e:
            print(f"生成哈希出错: {e}")
            return None

    def normalize_text(self, text):
        """统一文本格式，便于模糊匹配（去除所有空白并转小写）"""
        if not text:
            return ""
        return re.sub(r"\s+", "", str(text)).lower()

    def detect_hardware_accel(self):
        """检测显卡与可用加速后端（CUDA/OpenCL）"""
        adapter_names = []
        try:
            result = subprocess.run(
                ['wmic', 'path', 'win32_VideoController', 'get', 'Name'],
                capture_output=True,
                text=True,
                timeout=5
            )
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            adapter_names = [name for name in lines if name.lower() != 'name']
        except Exception:
            adapter_names = []

        has_cuda = False
        has_opencl = False
        reason = ""
        if cv2 is None or np is None:
            reason = "当前Python环境未安装opencv-python（cv2）或numpy"

        if cv2 is not None:
            try:
                has_opencl = bool(cv2.ocl.haveOpenCL())
            except Exception:
                has_opencl = False
            try:
                has_cuda = bool(cv2.cuda.getCudaEnabledDeviceCount() > 0)
            except Exception:
                has_cuda = False

        if has_cuda:
            backend = "CUDA"
        elif has_opencl:
            backend = "OpenCL"
        else:
            backend = "CPU"

        backend_usable = False
        if cv2 is not None and np is not None:
            try:
                if backend == "CUDA":
                    dummy = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
                    gpu_mat = cv2.cuda_GpuMat()
                    gpu_mat.upload(dummy)
                    _ = cv2.cuda.createGaussianFilter(
                        cv2.CV_8UC1, cv2.CV_8UC1, (3, 3), 0
                    ).apply(gpu_mat).download()
                    backend_usable = True
                elif backend == "OpenCL":
                    cv2.ocl.setUseOpenCL(True)
                    dummy = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
                    umat = cv2.UMat(dummy)
                    _ = cv2.GaussianBlur(umat, (3, 3), 0).get()
                    backend_usable = True
            except Exception:
                backend_usable = False

        if not reason and backend == "CPU":
            reason = "opencv已安装，但未检测到可用的OpenCL/CUDA后端（常见于opencv未启用CUDA）"
        if not reason and backend != "CPU" and not backend_usable:
            reason = f"检测到{backend}，但运行时自检失败，可能是驱动或OpenCV后端兼容问题"

        return {
            "adapters": adapter_names,
            "backend": backend,
            "usable": backend != "CPU" and cv2 is not None and np is not None and backend_usable,
            "reason": reason
        }

    def preprocess_image(self, image, use_accel=False):
        """图片预处理；启用加速时优先走OpenCL/CUDA路径"""
        if not use_accel or cv2 is None or np is None:
            gray = image.convert('L')
            enhancer = ImageEnhance.Contrast(gray)
            return enhancer.enhance(1.5)

        try:
            arr = np.array(image)
            if len(arr.shape) == 3:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            else:
                gray = arr

            if self.gpu_backend == "CUDA":
                gpu_mat = cv2.cuda_GpuMat()
                gpu_mat.upload(gray)
                gpu_eq = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gpu_mat)
                gray = gpu_eq.download()
                self.gpu_path_used_count += 1
            elif self.gpu_backend == "OpenCL":
                cv2.ocl.setUseOpenCL(True)
                umat = cv2.UMat(gray)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(umat).get()
                self.gpu_path_used_count += 1
            else:
                gray = cv2.equalizeHist(gray)

            return Image.fromarray(gray)
        except Exception:
            gray = image.convert('L')
            enhancer = ImageEnhance.Contrast(gray)
            return enhancer.enhance(1.5)
    
    def start_search(self):
        search_text = self.text_var.get().strip()
        if not search_text:
            messagebox.showerror("错误", "请输入要查找的文本")
            return
        if not self.target_folder:
            messagebox.showerror("错误", "请选择图片文件夹")
            return
        selected_engine = self.ocr_engine_var.get()
        if selected_engine == "Tesseract(兼容)":
            try:
                pytesseract.get_tesseract_version()
            except Exception:
                messagebox.showerror(
                    "错误",
                    "未检测到Tesseract OCR引擎，请先安装并确保 tesseract.exe 在系统PATH中，"
                    "或安装在 C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                )
                return
        else:
            if PaddleOCR is None or paddle is None:
                messagebox.showerror(
                    "错误",
                    "未安装PaddleOCR GPU依赖。\n请先执行:\n"
                    "pip install paddlepaddle-gpu paddleocr"
                )
                return

            if not paddle.is_compiled_with_cuda():
                messagebox.showwarning(
                    "提示",
                    "当前Paddle不是GPU版本，将继续运行但无法充分使用独显。\n"
                    "建议安装 paddlepaddle-gpu。"
                )

        self.gpu_accel_enabled = False
        self.gpu_backend = "CPU"
        self.gpu_path_used_count = 0
        if self.gpu_checkbox_var.get():
            accel_info = self.detect_hardware_accel()
            if accel_info["usable"]:
                self.gpu_accel_enabled = True
                self.gpu_backend = accel_info["backend"]
            else:
                adapter_text = "、".join(accel_info["adapters"]) if accel_info["adapters"] else "未检测到显卡"
                messagebox.showwarning(
                    "提示",
                    f"{adapter_text}\n{accel_info.get('reason', '未检测到可用的GPU加速后端（OpenCL/CUDA）')}\n将自动使用CPU继续运行。"
                )
        
        self.search_button.config(state=tk.DISABLED)
        if self.gpu_accel_enabled:
            self.status_var.set(f"正在搜索...（{selected_engine} | {self.gpu_backend}预处理加速）")
        else:
            self.status_var.set(f"正在搜索...（{selected_engine} | CPU）")
        self.progress_var.set(0)
        
        thread = threading.Thread(target=self.search_images, args=(search_text,))
        thread.daemon = True
        thread.start()
    
    def search_images(self, search_text):
        try:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
            image_files = []
            
            for root, dirs, files in os.walk(self.target_folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))
            
            if not image_files:
                self.status_var.set("未找到图片文件")
                self.search_button.config(state=tk.NORMAL)
                return
            
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            output_folder = os.path.join(desktop_path, "已查找图片")
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            os.makedirs(output_folder)
            
            found_count = 0
            total_files = len(image_files)
            need_update_index = False
            clean_search = self.normalize_text(search_text)
            ocr_mode = self.ocr_mode_var.get()
            worker_text = (self.worker_var.get() or "").strip()
            if worker_text:
                worker_count = max(1, int(worker_text))
            else:
                worker_count = max(1, min(8, os.cpu_count() or 4))
            selected_engine = self.ocr_engine_var.get()

            uncached_files = []
            processed_count = 0
            for image_path in image_files:
                try:
                    image_hash = self.get_image_hash(image_path)
                    cached_text = self.image_index.get(image_hash, "") if image_hash else ""
                    if cached_text:
                        if clean_search in self.normalize_text(cached_text):
                            shutil.copy2(image_path, output_folder)
                            found_count += 1
                    else:
                        uncached_files.append((image_path, image_hash))
                except Exception as e:
                    print(f"处理缓存阶段失败 {image_path}: {e}")
                processed_count += 1
                progress = processed_count / total_files * 100
                self.progress_var.set(progress)
                self.status_var.set(f"缓存检索中: {processed_count}/{total_files}")
                self.root.update_idletasks()

            def ocr_worker(path_hash):
                image_path, image_hash = path_hash
                text = self.ocr_image(image_path, mode=ocr_mode)
                normalized = self.normalize_text(text)
                matched = clean_search in normalized
                return image_path, image_hash, text, matched

            if uncached_files:
                self.status_var.set(f"OCR处理中(并行{worker_count}): 0/{len(uncached_files)}")
                self.root.update_idletasks()

                runtime_workers = worker_count
                with concurrent.futures.ThreadPoolExecutor(max_workers=runtime_workers) as executor:
                    futures = [executor.submit(ocr_worker, item) for item in uncached_files]
                    for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                        try:
                            image_path, image_hash, text, matched = future.result()
                            if image_hash and text.strip():
                                self.image_index[image_hash] = text
                                need_update_index = True
                            if matched:
                                shutil.copy2(image_path, output_folder)
                                found_count += 1
                        except Exception as e:
                            print(f"OCR任务失败: {e}")

                        processed_count += 1
                        progress = processed_count / total_files * 100
                        self.progress_var.set(progress)
                        self.status_var.set(
                            f"OCR处理中(并行{runtime_workers}): {i}/{len(uncached_files)}"
                        )
                        self.root.update_idletasks()
            
            if need_update_index or (uncached_files and not os.path.exists(self.index_file)):
                self.save_index()
            
            if found_count > 0:
                messagebox.showinfo("完成", f"找到 {found_count} 张包含文本的图片，已保存到桌面的'已查找图片'文件夹")
                self.status_var.set(
                    f"完成: 找到 {found_count} 张图片 | GPU预处理次数: {self.gpu_path_used_count}"
                )
            else:
                messagebox.showinfo("完成", "未找到包含指定文本的图片")
                self.status_var.set(
                    f"完成: 未找到匹配图片 | GPU预处理次数: {self.gpu_path_used_count}"
                )
                if os.path.exists(output_folder) and not os.listdir(output_folder):
                    os.rmdir(output_folder)
                    
        except Exception as e:
            messagebox.showerror("错误", f"搜索过程中出错: {str(e)}")
            self.status_var.set("错误")
        finally:
            self.search_button.config(state=tk.NORMAL)
            self.progress_var.set(100)
    
    def ocr_image(self, image_path, mode="平衡"):
        """使用OCR识别图片中的文本"""
        if self.ocr_engine_var.get() == "PaddleOCR-GPU(高性能)":
            paddle_text = self.ocr_image_paddle(image_path)
            if paddle_text.strip():
n                return paddle_text
            fallback_mode = "极速" if mode in ("极速", "平衡") else "精准"
            return self.ocr_image_tesseract(image_path, mode=fallback_mode)

        return self.ocr_image_tesseract(image_path, mode=mode)

    def ocr_image_tesseract(self, image_path, mode="平衡"):
        """使用Tesseract识别图片中的文本"""
        try:
            image = Image.open(image_path)

            if mode == "极速":
                gray = self.preprocess_image(image, use_accel=self.gpu_accel_enabled)
                return pytesseract.image_to_string(gray, lang='chi_sim+eng', config='--oem 1 --psm 6')

            image = self.preprocess_image(image, use_accel=self.gpu_accel_enabled)

            if mode == "平衡":
                configs = ['--oem 1 --psm 6', '--oem 1 --psm 11']
            else:
                configs = ['--oem 1 --psm 6', '--oem 1 --psm 3', '--oem 1 --psm 11']

            all_text = []
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image, lang='chi_sim+eng', config=config)
                    if text.strip():
                        all_text.append(text)
                except Exception:
                    pass

            if all_text:
                return ' '.join(all_text)

            return pytesseract.image_to_string(image, lang='chi_sim+eng')
        except Exception as e:
            print(f"OCR识别出错: {e}")
            return ""

    def get_paddle_ocr_for_thread(self):
        """获取当前线程对应的PaddleOCR实例（线程隔离，提升并发吞吐）"""
        ocr_instance = getattr(self._thread_local, "paddle_ocr", None)
        if ocr_instance is not None:
            return ocr_instance
        use_gpu = bool(paddle is not None and paddle.is_compiled_with_cuda())
        try:
            ocr_instance = PaddleOCR(
                use_angle_cls=False,
                lang='ch',
                use_gpu=use_gpu,
                show_log=False
            )
        except Exception as e:
            if "show_log" in str(e):
                ocr_instance = PaddleOCR(
                    use_angle_cls=False,
                    lang='ch',
                    use_gpu=use_gpu
                )
            else:
                raise
        self._thread_local.paddle_ocr = ocr_instance
        return ocr_instance

    def ocr_image_paddle(self, image_path):
        """使用PaddleOCR识别文本（GPU优先）"""
        try:
            ocr_engine = self.get_paddle_ocr_for_thread()
            result = ocr_engine.ocr(image_path, cls=False)
            if result is None:
                return ""
            texts = self.extract_paddle_texts(result)
            return " ".join(texts)
        except Exception as e:
            if not self._paddle_error_logged:
                print(f"PaddleOCR识别出错(仅显示一次): {e}")
                self._paddle_error_logged = True
            return ""

    def extract_paddle_texts(self, result):
        """兼容多版本PaddleOCR返回结构，提取文本列表"""
        texts = []

        def walk(node):
            if node is None:
                return

            if isinstance(node, dict):
                rec_texts = node.get("rec_texts")
                if isinstance(rec_texts, list):
                    for t in rec_texts:
                        tv = str(t).strip()
                        if tv:
                            texts.append(tv)
                for v in node.values():
                    walk(v)
                return

            if isinstance(node, str):
                tv = node.strip()
                if tv:
                    texts.append(tv)
                return

            if isinstance(node, (list, tuple)):
                if len(node) >= 2 and isinstance(node[1], (list, tuple)) and len(node[1]) >= 1:
                    maybe_text = node[1][0]
                    if isinstance(maybe_text, str):
                        tv = maybe_text.strip()
                        if tv:
                            texts.append(tv)
                for item in node:
                    walk(item)

        walk(result)
        unique = []
        seen = set()
        for t in texts:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageTextSearchApp(root)
    root.mainloop()