import os
import random
import shutil

def remain_30(root_dir = '../data/ModelNet40_180_tmp',new_dataset_dir = '../data/ModelNet_random_30'):
    """
    root_dir:原始数据文件路径
    new_dataset_dir:新数据集存储路径
    """

    # 遍历每个类别
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        
        # 遍历每个实例
        for instance in os.listdir(category_path):
            instance_path = os.path.join(category_path, instance)
            
            # 获取该实例下所有图片
            images = [os.path.join(instance_path, img) for img in os.listdir(instance_path)]
            
            # 随机选取30张图片
            selected_images = random.sample(images, 30)
            
            # 创建新数据集对应的目录结构
            new_instance_path = os.path.join(new_dataset_dir, category, instance)
            os.makedirs(new_instance_path, exist_ok=True)
            
            # 将选中的图片复制到新目录
            for img in selected_images:
                shutil.copy(img, new_instance_path)

def dataset_process(dataset_root = '../data/ModelNet_random_30',output_root = '../data/ModelNet_random_30_final', num_of_know = 20):
    
    """
    dataset_root: 输入文件夹根目录
    output_root: 输出文件夹根目录
    num_of_know: 已知类别数量
    """
    # 分为已知和未知两部分
    known_classes_dir = os.path.join(output_root, "DS")  # ../data/ModelNet_random_30_final/DS
    unknown_classes_dir = os.path.join(output_root, "DU")

    #创建目录
    os.makedirs(known_classes_dir, exist_ok = True)
    os.makedirs(unknown_classes_dir, exist_ok = True)

    # 将所有文件夹名称取出
    all_classes = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    random.shuffle(all_classes)

    known_classes = all_classes[:num_of_know]
    unknown_classes = all_classes[num_of_know:]

    # 处理已知类别
    known_train_dir = os.path.join(known_classes_dir, 'train')  # ../data/ModelNet_random_30_final/DS/train
    known_retrieeval_dir = os.path.join(known_classes_dir, 'retrieval')
    os.makedirs(known_train_dir, exist_ok=True)
    os.makedirs(known_retrieeval_dir, exist_ok=True)

    for cls in known_classes:
        class_dir = os.path.join(dataset_root, cls) #  ../data/ModelNet_random_30'/plane
        split_instance(class_dir, known_train_dir, known_retrieeval_dir)

    # 处理未知类别
    unknown_train_dir = os.path.join(unknown_classes_dir, 'train')
    unknown_retrieeval_dir = os.path.join(unknown_classes_dir, 'retrieval')
    os.makedirs(unknown_train_dir, exist_ok=True)
    os.makedirs(unknown_retrieeval_dir, exist_ok=True)

    for cls in unknown_classes:
        class_dir = os.path.join(dataset_root, cls)
        split_instance(class_dir, unknown_train_dir, unknown_retrieeval_dir)



def split_instance(class_dir = "../data/ModelNet_random_30/plane", output_train_dir = "../data/ModelNet_random_30_final/DS/train", output_retrieval_dir = "../data/ModelNet_random_30_final/DS/retrieval", train_ration = 0.8):
    instances = [d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir,d))]
    random.shuffle(instances)

    train_size = int(len(instances) * train_ration)
    train_instances = instances[:train_size]
    retrieval_instances = instances[train_size:]
    print(instances)

    for instance in train_instances:
        src_instance_dir = os.path.join(class_dir, instance) #"../data/ModelNet_random_30'/plane/instance1
        dst_instance_dir = os.path.join(output_train_dir, os.path.basename(class_dir), instance) # ../data/ModelNet_random_30_final/DS/train/plane/instance1

        os.makedirs(os.path.dirname(dst_instance_dir), exist_ok = True)
        shutil.copytree(src_instance_dir,dst_instance_dir)
    
    for instance in retrieval_instances:
        src_instance_dir = os.path.join(class_dir, instance)
        dst_instance_dir = os.path.join(output_retrieval_dir, os.path.basename(class_dir), instance) 

        os.makedirs(os.path.dirname(dst_instance_dir), exist_ok = True)
        shutil.copytree(src_instance_dir,dst_instance_dir)

def select_query_instance():
    # 数据根目录
    root_dir = "/home/xyzhang/project/Distillation/data/ModelNet_random_30_final/DS/retrieval"

    # 输出query文件夹路径
    query_dir = "/home/xyzhang/project/Distillation/data/ModelNet_random_30_final/DS/query"
    os.makedirs(query_dir, exist_ok=True)
    
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        if os.path.isdir(category_path):
            instances =  [os.path.join(category_path, instance) for instance in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, instance))]

            if len(instances) < 4: 
                print("类别实例数量不足，跳过该类别:", category)
                continue
            # 随机选择两个实例
            selected_instances = random.sample(instances, 4)

            target_category_dir = os.path.join(query_dir, category)
            os.makedirs(target_category_dir, exist_ok=True)

            for instance in selected_instances:
                instance_name = os.path.basename(instance)
                target_instance_dir = os.path.join(target_category_dir, instance_name)
                # os.makedirs(target_instance_dir, exist_ok=True)
                print(instance, target_category_dir)
                shutil.move(instance, target_category_dir)
                print(f"已复制实例 {instance} 到 {target_instance_dir}")

def move_query():

    # 定义源目录和目标目录
    source_dir = "/home/xyzhang/project/Distillation/data/ModelNet_random_30_final/DS/query/"
    target_dir = "/home/xyzhang/project/Distillation/data/ModelNet_random_30_final/DS/query/"

    # 遍历源目录下的所有子文件夹
    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        
        # 确保是文件夹
        if os.path.isdir(category_path):
            # 遍历类别文件夹中的所有文件
            for instance in os.listdir(category_path):
                instance_path = os.path.join(category_path, instance)
                
                # 构造目标文件名（类别名_实例名）
                target_instance_name = instance
                target_instance_path = os.path.join(target_dir, target_instance_name)
                
                # 移动文件
                shutil.move(instance_path, target_instance_path)
                print(f"Moved: {instance_path} -> {target_instance_path}")
    print("All instances have been moved successfully.")

def create_txt():
    # 定义源目录
    source_dir = "/home/xyzhang/project/Distillation/data/ModelNet_random_30_final/DS/target/"

    # 定义输出的txt文件路径
    output_file_path = "/home/xyzhang/project/Distillation/data/ModelNet_random_30_final/DS/target_label.txt"
    # 打开输出文件
    with open(output_file_path, "w") as output_file:

        for instance in os.listdir(source_dir):
            # 构造实例的完整路径
            instance_path = os.path.join(source_dir, instance)
            output_file.write(f"{instance},{instance[:-5]}\n")

    print(f"所有实例及其类别已成功写入到 {output_file_path}")
# dataset_process()
# select_query_instance()
# move_query()
create_txt()