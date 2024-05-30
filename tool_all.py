import os
import shutil

dataset_names = ["coffee_martini"]
paths = ["imgs_test_all"]

for dataset_name in dataset_names:
    ################ MAKE CHANGES HERE #################
    # dataset_name = "flame_steak"
    for path in paths:
        data_path = "C:\\Users\\xingyi.li\\Desktop\\s-dyrf"
        data_path = os.path.join(data_path, dataset_name, path)
        gen = os.path.join(data_path, dataset_name + "_gen")
        if not os.path.exists(gen):
            os.mkdir(gen)
        train = os.path.join(data_path, dataset_name + "_train")
        if not os.path.exists(train):
            os.mkdir(train)
        flow = os.path.join(data_path, dataset_name + "_flow")
        if not os.path.exists(flow):
            os.mkdir(flow)
        mask_dir = os.path.join(data_path, "mask")

        flowFwdDir = os.path.join(flow, "flow_fwd")
        flowBwdDir = os.path.join(flow, "flow_bwd")

        if path == "imgs_test_all":
            FIRST = 0                      # number of the first PNG file in the input folder
            LAST = 299                     # number of the last PNG file in the input folder
        else:
            FIRST = 0                      # number of the first PNG file in the input folder
            LAST = 119                     # number of the last PNG file in the input folder

        inputFileFormat = "%03d"    # name of input files, e.g., %03d if files are named 001.png, 002.png
        inputFileExt = "png"            # extension of input files (without .), e.g., png, jpg
        filtered_dir = os.path.join(data_path, "input_filtered")
        input_filtered = os.path.join(filtered_dir, inputFileFormat + ".png")  # path to the result filtered sequence
        maskFiles = os.path.join(mask_dir, inputFileFormat + ".png")

        gdisko_gauss_r10_s10_dir = os.path.join(data_path, "input_gdisko_gauss_r10_s10")    # path to the result gauss r10 s10 sequence
        gdisko_gauss_r10_s15_dir = os.path.join(data_path, "input_gdisko_gauss_r10_s15")    # path to the result gauss r10 s15 sequence
        gdisko_gauss_r10_s10_files = gdisko_gauss_r10_s10_dir + "/" + inputFileFormat + ".png" 
        gdisko_gauss_r10_s15_files = gdisko_gauss_r10_s15_dir + "/" + inputFileFormat + ".png" 
        ####################################################

        # disflow
        inputFiles = os.path.join(data_path, "test" + inputFileFormat + "." + inputFileExt)
        flwFwdFile = flowFwdDir + "/" + inputFileFormat + ".A2V2f"
        flwBwdFile = flowBwdDir + "/" + inputFileFormat + ".A2V2f"
        firstFrame = FIRST+1
        lastFrame  = LAST
        frameStep  = +1
        if not os.path.exists(flowFwdDir):
            os.mkdir(flowFwdDir)

            for frame in range(firstFrame,lastFrame+frameStep,frameStep):
                os.system("disflow %s %s %s"%(inputFiles%(frame),inputFiles%(frame-frameStep),flwFwdFile%(frame)))

        firstFrame = LAST-1
        lastFrame  = FIRST
        frameStep  = -1
        if not os.path.exists(flowBwdDir):
            os.mkdir(flowBwdDir)

            for frame in range(firstFrame,lastFrame+frameStep,frameStep):
                os.system("disflow %s %s %s"%(inputFiles%(frame),inputFiles%(frame-frameStep),flwBwdFile%(frame)))

        # bilateralAdv
        if not os.path.exists(filtered_dir):
            firstFrame = FIRST
            lastFrame= LAST
            frameStep = +1

            os.makedirs(os.path.dirname(input_filtered), exist_ok=True)

            for frame in range(firstFrame,lastFrame+frameStep,frameStep):  	
                filter = "bilateralAdv.exe "+inputFiles+" "+flwFwdFile+" "+flwBwdFile+(" %d "%(frame))+" 15 16 "+(input_filtered%(frame))
                #print(filter)
                os.system(filter)

        # generate mask
        import shutil

        if not os.path.exists(mask_dir): 
            # path to source directory
            src_dir = filtered_dir

            # path to destination directory
            dest_dir = mask_dir

            # getting all the files in the source directory
            files = os.listdir(filtered_dir)

            shutil.copytree(src_dir, dest_dir)

            from PIL import Image, ImageEnhance
            import glob
            images = glob.glob(dest_dir+'/*.png')

            for image in images:
                img_path = os.path.join(dest_dir, image)
                if os.path.isfile(img_path):
                    img = Image.open(img_path)
                    contrast = ImageEnhance.Contrast(img)
                    factor = 0
                    img = contrast.enhance(factor)
                    enhancer = ImageEnhance.Brightness(img)
                    factor2 = 1000 #brightens the image
                    img = enhancer.enhance(factor2)
                    img.save(img_path)

        if not os.path.exists(gdisko_gauss_r10_s10_dir):
            os.mkdir(gdisko_gauss_r10_s10_dir)
            
        if not os.path.exists(gdisko_gauss_r10_s15_dir):
            os.mkdir(gdisko_gauss_r10_s15_dir)

        # gauss
        frameFirst = FIRST
        frameLast = LAST

        masks_str = ""
        masks_list_dir = os.listdir(mask_dir)
        for mask in masks_list_dir:
            masks_str += mask.replace(".png", "").replace(".jpg", "")
            masks_str += " "


        os.system(f"gauss.exe {maskFiles} {flwFwdFile} {flwBwdFile} {frameFirst} {frameLast} {len(masks_list_dir)} {masks_str} 10 10 {gdisko_gauss_r10_s10_files}")
        os.system(f"gauss.exe {maskFiles} {flwFwdFile} {flwBwdFile} {frameFirst} {frameLast} {len(masks_list_dir)} {masks_str} 10 15 {gdisko_gauss_r10_s15_files}")

        # copy files to train and gen
        # _train
        dest_input_filtered = os.path.join(train, "input_filtered")
        dest_input_gdisko_gauss_r10_s10 = os.path.join(train, "input_gdisko_gauss_r10_s10")
        dest_input_gdisko_gauss_r10_s15 = os.path.join(train, "input_gdisko_gauss_r10_s15")
        dest_mask = os.path.join(train, "mask")
        dest_output = os.path.join(train, "output")
        os.makedirs(dest_input_filtered, exist_ok=True)
        os.makedirs(dest_input_gdisko_gauss_r10_s10, exist_ok=True)
        os.makedirs(dest_input_gdisko_gauss_r10_s15, exist_ok=True)
        os.makedirs(dest_mask, exist_ok=True)
        os.makedirs(dest_output, exist_ok=True)

        shutil.copy(os.path.join(filtered_dir, "000.png"), os.path.join(dest_input_filtered, "000.png"))
        shutil.copy(os.path.join(gdisko_gauss_r10_s10_dir, "000.png"), os.path.join(dest_input_gdisko_gauss_r10_s10, "000.png"))
        shutil.copy(os.path.join(gdisko_gauss_r10_s15_dir, "000.png"), os.path.join(dest_input_gdisko_gauss_r10_s15, "000.png"))
        shutil.copy(os.path.join(mask_dir, "000.png"), os.path.join(dest_mask, "000.png"))

        # _gen
        dest_input_filtered = os.path.join(gen, "input_filtered")
        dest_input_gdisko_gauss_r10_s10 = os.path.join(gen, "input_gdisko_gauss_r10_s10")
        dest_input_gdisko_gauss_r10_s15 = os.path.join(gen, "input_gdisko_gauss_r10_s15")
        dest_mask = os.path.join(gen, "mask")

        shutil.copytree(filtered_dir, dest_input_filtered)
        shutil.copytree(gdisko_gauss_r10_s10_dir, dest_input_gdisko_gauss_r10_s10)
        shutil.copytree(gdisko_gauss_r10_s15_dir, dest_input_gdisko_gauss_r10_s15)
        shutil.copytree(mask_dir, dest_mask)