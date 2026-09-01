# 环境自检：打印 Python / torch / CUDA / 依赖版本
$python = "E:\ScaleLab\CS231n\project\.envs\cs231n\python.exe"
& $python -c "import sys, torch; print('Python', sys.version.split()[0]); print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); import numpy, scipy, sklearn, matplotlib, PIL, imageio, thop, pyarrow, cv2; print('core deps OK')"
