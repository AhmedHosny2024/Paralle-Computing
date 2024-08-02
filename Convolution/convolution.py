import sys
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
#torch.manual_seed(100)  # Set a random seed

if __name__ == '__main__':
    # get the first argument
    if len(sys.argv) !=5:
        print("Usage: python convolution.py <input> <output> <batch>")
        sys.exit(1)
    # check if the input directory exists
    if not os.path.exists(sys.argv[1]):
        print("Input directory does not exist")
        sys.exit(1)
    # ckeck if the output directory exists
    if not os.path.exists(sys.argv[2]):
        print("Output directory does not exist")
        sys.exit(1)
    # check if the filter file exists
    if not os.path.exists(sys.argv[4]):
        print("Filter file does not exist")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    batch = int(sys.argv[3])
    filter_path = sys.argv[4]
    mask=[]
    filter_size = 0
    index=0
    with open(filter_path, 'r') as f:
        for line in f:
            if index==0:
                filter_size = int(line)
                index+=1
                continue
            mask.append([float(i) for i in line.split()])
    # make the kernel tensor
    mask = torch.tensor(mask)
    if not os.path.exists(input_dir):
        print("Input directory does not exist")
        sys.exit(1)
    all_images = []
    images_names = []
    # for all the images in the input directory read and store them
    for image in os.listdir(input_dir):
        img = Image.open(input_dir+"/" + image)
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img)
        all_images.append(img) 
        images_names.append(image)

    # Define the convolutional layer
    conv = torch.nn.Conv2d(3, 1, kernel_size=filter_size, padding="same")
    mask = mask.repeat(3, 1, 1)
    mask = mask.unsqueeze(0)

    # set the weights of the convolutional layer to be the mask
    conv.weight.data = mask
    conv.bias.data = torch.zeros_like(conv.bias.data)

    # for all the images in the input directory apply the convolution using the batch size
    image_index=0
    for i in range(0, len(all_images), batch):
        images = all_images[i:i+batch]
        images = torch.stack(images)
        # apply the convolution
        convolved_images = conv(images)
        # save the image
        for idx, img in enumerate(convolved_images):
            img = transforms.ToPILImage()(img)
            img.save(output_dir+"/"+ f'{images_names[image_index]}')
            image_index+=1
    print("Convolution completed")