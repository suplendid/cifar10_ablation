# cifar10_ablation
vgg16 and resnet18 experiment in cifar10

there are 12 parameters in the code

    parser.add_argument("--model", default="vgg", type=str, help='model name(vgg or resnet)')
    parser.add_argument("--valid", default=1, type=int, help='apply validation set from training set or not')
    parser.add_argument("--less_data", default=1, type=int, help='apply imbalanced training set or not' )
    parser.add_argument("--batch_size", default=128, type=int, help='batch size')
    parser.add_argument("--weight", default=1, type=int, help='apply class weight or not')
    parser.add_argument("--valid_rate", default=0.2, type=float, help='ratio of validation set compared to original training set')
    parser.add_argument("--ROS", default=1, type=int, help='apply random over sampling or not')
    parser.add_argument("--epochs", default=200, type=int, help='epochs for training')
    parser.add_argument("--c", default=1, type=int, help='apply transform RandomCrop or not')
    parser.add_argument("--f", default=1, type=int, help='apply transform RandomFlip or not')
    parser.add_argument("--e", default=1, type=int, help='apply transform RandomErasing or not')
    parser.add_argument("--cutmix", default=0, type=int, help='apply cutmix or not')

You can do any ablations (total 2^12) you like by running run.py
    
    python run.py

For a single training scenario use the following code
    
    python train.py --model vgg --valid 1 --less_data 1 --weight 1 --ROS 1 --c 1 --f 1 --e 1 --cutmix 0 --batch_size 128 --epochs 200 --valid_rate 0.2
