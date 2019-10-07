tr_anno_path = args.train_ann_dir
tr_img_dir = args.train_image_dir
tr_mask_dir = os.path.join(dataset_dir, "trainmask2014")

val_anno_path = args.val_ann_dir
val_img_dir = args.val_image_dir
val_mask_dir = os.path.join(dataset_dir, "valmask2014")

datasets = [
    (val_anno_path, val_img_dir, val_mask_dir, "COCO_val", "val"),  # it is important to have 'val' in validation dataset name, look for 'val' below
    (tr_anno_path, tr_img_dir, tr_mask_dir, "COCO", "train")
]


joint_all = []
# tr_hdf5_path = os.path.join(dataset_dir, "train_dataset_2014.h5")
# val_hdf5_path = os.path.join(dataset_dir, "val_dataset_2014.h5")

val_size = 2645 # size of validation set


def process():
    count = 0
    for _, ds in enumerate(datasets):

        anno_path = ds[0]
        images_dir = ds[1]
        masks_dir = ds[2]
        dataset_type = ds[3]
        train_val_mode = ds[4]

        coco = COCO(anno_path)
        ids = list(coco.imgs.keys())
        max_images = len(ids)

        dataset_count = 0
        for image_index, img_id in enumerate(ids):
            ann_ids = coco.getAnnIds(imgIds=img_id)
            img_anns = coco.loadAnns(ann_ids)

            numPeople = len(img_anns)
            image = coco.imgs[img_id]
            h, w = image['height'], image['width']

            print("Image ID ", img_id)

            all_persons = []

            for p in range(numPeople):

                pers = dict()

                person_center = [img_anns[p]["bbox"][0] + img_anns[p]["bbox"][2] / 2,
                                 img_anns[p]["bbox"][1] + img_anns[p]["bbox"][3] / 2]

                pers["objpos"] = person_center
                pers["bbox"] = img_anns[p]["bbox"]
                pers["segment_area"] = img_anns[p]["area"]
                pers["num_keypoints"] = img_anns[p]["num_keypoints"]

                anno = img_anns[p]["keypoints"]

                pers["joint"] = np.zeros((17, 3))
                for part in range(17):
                    pers["joint"][part, 0] = anno[part * 3]
                    pers["joint"][part, 1] = anno[part * 3 + 1]

                    if anno[part * 3 + 2] == 2:
                        pers["joint"][part, 2] = 1
                    elif anno[part * 3 + 2] == 1:
                        pers["joint"][part, 2] = 0
                    else:
                        pers["joint"][part, 2] = 2

                pers["scale_provided"] = img_anns[p]["bbox"][3] / 368

                all_persons.append(pers)


            main_persons = []
            prev_center = []

            for pers in all_persons:

                # skip this person if parts number is too low or if
                # segmentation area is too small
                if pers["num_keypoints"] < 5 or pers["segment_area"] < 32 * 32:
                    continue

                person_center = pers["objpos"]

                # skip this person if the distance to exiting person is too small
                flag = 0
                for pc in prev_center:
                    a = np.expand_dims(pc[:2], axis=0)
                    b = np.expand_dims(person_center, axis=0)
                    dist = cdist(a, b)[0]
                    if dist < pc[2]*0.3:
                        flag = 1
                        continue

                if flag == 1:
                    continue

                main_persons.append(pers)
                prev_center.append(np.append(person_center, max(img_anns[p]["bbox"][2], img_anns[p]["bbox"][3])))


            for p, person in enumerate(main_persons):

                joint_all.append(dict())

                joint_all[count]["dataset"] = dataset_type

                if image_index < val_size and 'val' in dataset_type:
                    isValidation = 1
                else:
                    isValidation = 0

                joint_all[count]["isValidation"] = isValidation

                joint_all[count]["img_width"] = w
                joint_all[count]["img_height"] = h
                joint_all[count]["image_id"] = img_id
                joint_all[count]["annolist_index"] = image_index

                # set image path
                joint_all[count]["img_paths"] = os.path.join(images_dir, 'COCO_%s2014_%012d.jpg' %(train_val_mode,img_id))
                # joint_all[count]["img_paths"] = os.path.join(images_dir, '%012d.jpg' % img_id)
                joint_all[count]["mask_miss_paths"] = os.path.join(masks_dir,
                                                                   'mask_miss_%012d.png' % img_id)
                joint_all[count]["mask_all_paths"] = os.path.join(masks_dir,
                                                                  'mask_all_%012d.png' % img_id)

                # set the main person
                joint_all[count]["objpos"] = main_persons[p]["objpos"]
                joint_all[count]["bbox"] = main_persons[p]["bbox"]
                joint_all[count]["segment_area"] = main_persons[p]["segment_area"]
                joint_all[count]["num_keypoints"] = main_persons[p]["num_keypoints"]
                joint_all[count]["joint_self"] = main_persons[p]["joint"]
                joint_all[count]["scale_provided"] = main_persons[p]["scale_provided"]

                # set other persons
                joint_all[count]["joint_others"] = []
                joint_all[count]["scale_provided_other"] = []
                joint_all[count]["objpos_other"] = []
                joint_all[count]["bbox_other"] = []
                joint_all[count]["segment_area_other"] = []
                joint_all[count]["num_keypoints_other"] = []

                lenOthers = 0
                for ot, operson in enumerate(all_persons):

                    if person is operson:
                        assert not "people_index" in joint_all[count], "several main persons? couldn't be"
                        joint_all[count]["people_index"] = ot
                        continue

                    if operson["num_keypoints"]==0:
                        continue

                    joint_all[count]["joint_others"].append(all_persons[ot]["joint"])
                    joint_all[count]["scale_provided_other"].append(all_persons[ot]["scale_provided"])
                    joint_all[count]["objpos_other"].append(all_persons[ot]["objpos"])
                    joint_all[count]["bbox_other"].append(all_persons[ot]["bbox"])
                    joint_all[count]["segment_area_other"].append(all_persons[ot]["segment_area"])
                    joint_all[count]["num_keypoints_other"].append(all_persons[ot]["num_keypoints"])

                    lenOthers += 1

                assert "people_index" in joint_all[count], "No main person index"
                joint_all[count]["numOtherPeople"] = lenOthers
                count += 1
                dataset_count += 1
                