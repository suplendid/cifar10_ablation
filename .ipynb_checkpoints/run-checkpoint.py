import subprocess

model = ['resnet','vgg']
valid = [1,0]
less_data = [1, 0]
weight = [1, 0]
ROS = [1, 0]
crop = [1,0]
flip = [1,0]
erase = [1,0]
cutmix = [1,0]

for m in model:
    for l in less_data:
        for v in valid:
            for w in weight:
                for r in ROS:
                    for c in crop:
                        for f in flip:
                            for e in erase:
                                if e == 1:   ### RandomErase does not run with cutmix
                                    subprocess.call(["env","CUDA_VISIBLE_DEVICES=0","python","train.py","--model",f"{m}","--valid",f"{v}","--less_data",f"{l}","--weight",f"{w}","--ROS",f"{r}","--c",f"{c}","--f",f"{f}","--e",f"{e}","--cutmix",f"0"])
                                else:
                                    for cut in cutmix:
                                        subprocess.call(["env","CUDA_VISIBLE_DEVICES=0","python","train.py","--model",f"{m}","--valid",f"{v}","--less_data",f"{l}","--weight",f"{w}","--ROS",f"{r}","--c",f"{c}","--f",f"{f}","--e",f"{e}","--cutmix",f"{cut}"])
# data augmentation ablation
#for l in less_data:
#    for c in crop:
#        for f in flip:
#            for e in erase:
#                subprocess.call(["env","CUDA_VISIBLE_DEVICES=3","python","train.py","--valid",f"{v}","--less_data",f"{l}","--weight",f"{w}","--ROS",f"{r}","--c",f"{c}","--f",f"{f}","--e",f"{e}"])

# cutmix ablation
#for l in less_data:
#    for m in model:
#        for e in erase:
#            if e == 1:
#                subprocess.call(["env","CUDA_VISIBLE_DEVICES=1","python","train.py","--model",f"{m}","--valid",f"{v}","--less_data",f"{l}","--weight",f"{w}","--ROS",f"{r}","--c",f"{c}","--f",f"{f}","--e",f"{e}","--cutmix",f"0"])
#            else:
#                for cut in cutmix:
#                    subprocess.call(["env","CUDA_VISIBLE_DEVICES=1","python","train.py","--model",f"{m}","--valid",f"{v}","--less_data",f"{l}","--weight",f"{w}","--ROS",f"{r}","--c",f"{c}","--f",f"{f}","--e",f"{e}","--cutmix",f"{cut}"])

# techique ablation
#for v in valid:
#    for l in less_data:
#        for w in weight:
#            for r in ROS:
#                subprocess.call(["python","train.py","--valid",f"{v}","--less_data",f"{l}","--weight",f"{w}","--ROS",f"{r}"])