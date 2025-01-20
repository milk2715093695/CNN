import torch
import tkinter as tk
from typing import List
from PIL import Image, ImageDraw, ImageTk
from src.models.CNN import CNN_MNIST as CNN
from src.data.transforms import transform_data

class HandwrittenDigitRecognition:
    def __init__(self, root: tk.Tk, model: torch.nn.Module, images: List[Image.Image], /, canvas_width: int = 1200) -> None:
        # 初始化尺寸
        self._initialize_dimensions(root, canvas_width)

        # 初始化变量
        self.model = model
        self.images = images

        # 设置窗口
        self._setup_window()

        # 创建画布和绘图区域
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()
        self.paint_canvas = self._create_paint_canvas()

        # 使用Pillow绘制
        self.image = Image.new("RGB", (self.square_size, self.square_size), color="black")
        self.draw = ImageDraw.Draw(self.image)

        # 初始化鼠标状态
        self.is_drawing = False
        self.last_x, self.last_y = None, None

        # 默认笔刷粗细
        self.drawing_width = self.drawing_width_default

        # 绑定鼠标事件
        self.paint_canvas.bind("<Button-1>", self.start_drawing)
        self.paint_canvas.bind("<B1-Motion>", self.draw_line)
        self.paint_canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # 创建按钮
        self.clear_button = self._create_button("清除", self.clear_canvas, self.canvas_width // 2, (self.canvas_height - self.square_size) // 2)
        self.display_button = self._create_button("展示", self.display_image, self.canvas_width // 2, self.canvas_height // 2)
        self.display_button = self._create_button("预测", self.predict, self.canvas_width // 2, (self.canvas_height + self.square_size) // 2)

        # 创建笔刷粗细控制滑动条
        self.brush_slider = self._create_slider()

    def _initialize_dimensions(self, root: tk.Tk, canvas_width: int) -> None:
        """初始化尺寸"""
        self.root = root
        self.canvas_width = canvas_width
        self.canvas_height = int(canvas_width * 7 / 12)
        self.square_size = int(canvas_width * 1 / 3)
        self.drawing_width_default = int(self.square_size * 1 / 16)
        self.drawing_width = self.drawing_width_default
        self.offset_horizontal = 0
        self.offset_vertical = -self.canvas_height // 14
        self.padding_horizontal = int(canvas_width * 1 / 24)
        self.padding_vertical = int(self.canvas_height * 1 / 14)

    def _setup_window(self) -> None:
        """设置窗口居中"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        position_left = int(screen_width / 2 - self.canvas_width / 2 + self.offset_horizontal)
        position_top = int(screen_height / 2 - self.canvas_height / 2 + self.offset_vertical)
        self.root.geometry(f"{self.canvas_width}x{self.canvas_height}+{position_left}+{position_top}")
        self.root.title("手写数字识别")

    def _create_paint_canvas(self) -> tk.Canvas:
        """创建绘画区域"""
        paint_canvas = tk.Canvas(self.root, width=self.square_size, height=self.square_size, bg="black")
        paint_canvas.place(x=self.padding_horizontal, y=(self.canvas_height - self.square_size) // 2)
        return paint_canvas

    def _create_button(self, text: str, command: callable, x: int, y: int) -> tk.Button:
        """创建按钮"""
        button = tk.Button(self.root, text=text, command=command, height=2, width=5, font=('Arial', 20, 'bold'))
        button.place(x=x, y=y)
        return button

    def _create_slider(self) -> tk.Scale:
        """创建控制笔刷粗细的滑动条"""
        slider = tk.Scale(self.root, from_=15, to=40, orient="horizontal", label="笔刷粗细", command=self.update_brush_width)
        slider.set(self.drawing_width_default)
        slider.place(x=self.padding_horizontal, y=(self.canvas_height + self.square_size) // 2 + self.padding_vertical)
        return slider

    def update_brush_width(self, value: str) -> None:
        """更新笔刷粗细"""
        self.drawing_width = int(value)

    def start_drawing(self, event: tk.Event) -> None:
        """鼠标按下事件，开始绘制"""
        self.is_drawing = True
        self.last_x, self.last_y = event.x, event.y

    def draw_line(self, event: tk.Event) -> None:
        """鼠标拖动事件，绘制线条"""
        if self.is_drawing:
            x, y = event.x, event.y
            self.paint_canvas.create_line(self.last_x, self.last_y, x, y, width=self.drawing_width, fill="white", capstyle=tk.ROUND)
            self.draw.line([self.last_x, self.last_y, x, y], fill="white", width=self.drawing_width)
            self.last_x, self.last_y = x, y

    def stop_drawing(self, event: tk.Event) -> None:
        """鼠标松开事件，停止绘制"""
        self.is_drawing = False

    def clear_canvas(self) -> None:
        """清除画布"""
        self.paint_canvas.delete("all")
        self.image = Image.new("RGB", (self.square_size, self.square_size), color="black")
        self.draw = ImageDraw.Draw(self.image)

    def display_image(self) -> None:
        """展示绘制的图像"""
        cropped_image = self._adjust_image()
        cropped_image = cropped_image.resize((self.square_size, self.square_size), Image.NEAREST)
        tk_image = ImageTk.PhotoImage(cropped_image)
        self.canvas.create_image(self.canvas_width - self.padding_horizontal - self.square_size // 2, self.canvas_height // 2, image=tk_image)
        self.canvas.tk_image = tk_image

    def _crop_image(self, aspect_ratio_threshold: float = 1 / 0.6) -> Image:
        """裁剪最小边界框"""
        bbox = self.image.getbbox()
        cropped_image = self.image.crop(bbox) if bbox else self.image
        width, height = cropped_image.size
        
        # 判断长宽比的范围
        if 1 / aspect_ratio_threshold < width / height < aspect_ratio_threshold:
            cropped_image = cropped_image.resize((20, 20))
        else:
            if width > height:
                new_height = int(20 * height / width)
                cropped_image = cropped_image.resize((20, new_height))
            else:
                new_width = int(20 * width / height)
                cropped_image = cropped_image.resize((new_width, 20))

            # 创建 20x20 的黑色背景
            final_image = Image.new("RGB", (20, 20), color="black")
            # 将 cropped_image 居中放置到 final_image 中
            paste_x = (20 - cropped_image.width) // 2
            paste_y = (20 - cropped_image.height) // 2
            final_image.paste(cropped_image, (paste_x, paste_y))
            
            cropped_image = final_image

        return cropped_image

    def _calculate_centroid(self, image: Image) -> tuple:
        """根据亮度加权计算质心"""
        pixels = image.load()
        width, height = image.size
        total_x = total_y = total_weight = 0

        # 遍历所有像素，计算加权质心
        for y in range(height):
            for x in range(width):
                brightness = pixels[x, y]
                if not (isinstance(brightness, int) or isinstance(brightness, float)):
                    r, g, b = brightness  # 获取RGB值
                    brightness = 0.2989 * r + 0.5870 * g + 0.1140 * b  # 计算亮度
                total_x += x * brightness
                total_y += y * brightness
                total_weight += brightness
        
        # 计算加权质心坐标
        if total_weight > 0:
            centroid_x = total_x // total_weight
            centroid_y = total_y // total_weight
            return (int(centroid_x), int(centroid_y))
        else:
            return (width // 2, height // 2)  # 如果没有亮度值，返回图像中心

    def _adjust_image(self) -> Image:
        """将裁剪图像的质心放置到28x28黑色背景图像的中间"""
        # 裁剪图像并计算质心
        cropped_image = self._crop_image()
        centroid_x, centroid_y = self._calculate_centroid(cropped_image)

        # 创建28x28的黑色背景
        background = Image.new("RGB", (28, 28), color="black")
        
        # 对其质心和中心
        offset_x = 14 - centroid_x
        offset_y = 14 - centroid_y

        offset_x = offset_x if offset_x <= 8 else 8
        offset_y = offset_y if offset_y <= 8 else 8
        
        background.paste(cropped_image, (offset_x, offset_y))
        
        return background
    
    def _get_image_tensor(self) -> torch.Tensor:
        """得到绘制的图像对应的 tensor"""
        grayscale_image = self._adjust_image().convert("L")  # 转为灰度图
        
        image_tensor = torch.tensor(list(grayscale_image.getdata()), dtype=torch.float32)
        
        image_tensor = image_tensor.view(28, 28)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # 增加两个维度，得到 (1, 1, 28, 28)

        return image_tensor

    def _predict(self) -> int:
        """预测函数（内部）"""
        # 获取输入
        input = self._get_image_tensor()
        input = transform_data(input)

        # 预测
        output = self.model(input)
        _, predicted = torch.max(output, 1)

        return predicted.item()
    
    def predict(self) -> None:
        """预测函数（按钮）"""
        res = self._predict()
        self.image = self.images[res]
        self.display_image()

if __name__ == "__main__":
    model = CNN()
    model.load_state_dict(torch.load("./models/MNIST.pth", weights_only=True))

    images = []
    for num in range(10):
        image_path = f"images/28_28/{num}.png"
        image = Image.open(image_path)
        images.append(image)

    root = tk.Tk()
    app = HandwrittenDigitRecognition(root, model, images)
    root.mainloop()