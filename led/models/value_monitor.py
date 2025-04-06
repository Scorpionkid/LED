import torch
import os


class ValueMonitor:
    def __init__(self, auto_save=True, save_freq=2000, save_dir="debug", threshold=5.0):
        super().__init__()
        self.auto_save = auto_save
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.last_abnormal_count = 0
        self.hooks = []
        self.threshold = threshold
        self.abnormal_modules = {}
        self.current_iter = 0

        # 确保保存目录存在
        if self.auto_save:
            os.makedirs(self.save_dir, exist_ok=True)

    def update_iter(self, iter_num):
        """更新迭代计数并自动保存"""
        self.current_iter = iter_num

        # 自动保存逻辑
        if self.auto_save:
            # 定期保存
            if iter_num % self.save_freq == 0:
                self.save_report(f"{self.save_dir}/report_{iter_num}.txt")

                # # 如果检测到新的异常值，立即保存
                # total_abnormal = sum(len(data['inputs']) + len(data['outputs'])
                #                 for data in self.abnormal_modules.values())

                # if total_abnormal > self.last_abnormal_count:
                #     self.save_report(f"{self.save_dir}/abnormal_{iter_num}.txt")
                #     self.last_abnormal_count = total_abnormal

    def register_hooks(self, model, prefix=''):
        """为模型的所有模块注册钩子"""
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # 使用方法引用而非 lambda 来避免闭包问题
            hook = module.register_forward_hook(self._make_hook(full_name))
            self.hooks.append(hook)

            # 递归处理子模块
            self.register_hooks(module, full_name)

    def _make_hook(self, name):
        """创建一个特定模块的钩子函数"""
        def hook_fn(module, inputs, outputs):
            return self.check_values(module, inputs, outputs, name)
        return hook_fn

    def check_values(self, module, inputs, outputs, name):
        """检查模块的输入和输出是否包含异常值"""
        # 检查输入
        if isinstance(inputs, tuple) and len(inputs) > 0:
            input_tensor = inputs[0]
            if isinstance(input_tensor, torch.Tensor):
                max_val = torch.max(torch.abs(input_tensor)).item()
                if max_val > self.threshold and self.current_iter % self.save_freq == 0:
                    print(f"异常输入检测 - 迭代: {self.current_iter}, 模块: {name}, 最大绝对值: {max_val:.4f}")
                    if name not in self.abnormal_modules:
                        self.abnormal_modules[name] = {'inputs': [], 'outputs': []}
                    self.abnormal_modules[name]['inputs'].append((self.current_iter, max_val))

        # 检查输出
        if isinstance(outputs, torch.Tensor):
            max_val = torch.max(torch.abs(outputs)).item()
            if max_val > self.threshold and self.current_iter % self.save_freq == 0:
                print(f"异常输出检测 - 迭代: {self.current_iter}, 模块: {name}, 最大绝对值: {max_val:.4f}")
                if name not in self.abnormal_modules:
                    self.abnormal_modules[name] = {'inputs': [], 'outputs': []}
                self.abnormal_modules[name]['outputs'].append((self.current_iter, max_val))

    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def save_report(self, path="debug/module_report.txt"):

        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            """保存当前的监控报告"""
            with open(path, "w") as f:
                f.write(f"异常值监控报告 (当前迭代: {self.current_iter})\n")
                f.write("="*50 + "\n\n")

                for module_name, data in self.abnormal_modules.items():
                    f.write(f"模块: {module_name}\n")
                    if data['inputs']:
                        f.write("  异常输入:\n")
                        for iter_num, value in data['inputs']:
                            f.write(f"    迭代 {iter_num}: {value:.4f}\n")
                    if data['outputs']:
                        f.write("  异常输出:\n")
                        for iter_num, value in data['outputs']:
                            f.write(f"    迭代 {iter_num}: {value:.4f}\n")
                    f.write("\n")
        except Exception as e:
            print(f"保存报告失败: {str(e)}")