
# loading any random image
train_path  = '/Users/muhammadgulfamtahir/PycharmProjects/brain_tumor_semantic/train'
test_path = '/Users/muhammadgulfamtahir/PycharmProjects/brain_tumor_semantic/test'
val_path  = '/Users/muhammadgulfamtahir/PycharmProjects/brain_tumor_semantic/valid'


train_images = [image for image in os.listdir(train_path) if image[-3:] =='jpg' ]
test_images = [image for image in os.listdir(test_path) if image[-3:] =='jpg' ]
val_images = [image for image in os.listdir(val_path) if image[-3:] =='jpg' ]
len(train_images),len(test_images),len(val_images)


train_annotations = glob.glob(os.path.join(train_path, '*.json'))
test_annotations = glob.glob(os.path.join(test_path, '*.json'))
val_annotations = glob.glob(os.path.join(val_path, '*.json'))

train_annotations = json.load(open(train_annotations[0]))
test_annotations = json.load(open(test_annotations[0]))
val_annotations = json.load(open(val_annotations[0]))



def visualize_random_images(n=5):
  # select n random images
  # use cv and plt to show these images
  indices = np.random.randint(0, len(train_annotations['images']), size=n)

  images =[train_annotations['images'][i] for i in indices ]

  annotations = [train_annotations['annotations'][i] for i in indices ]
  j=1
  plt.figure(figsize=(12, 4 * 2 * n))
  for img,ann in zip(images,annotations):
    plt.subplot(n,3,j)
    j+=1
    image = cv2.imread(train_path+ "/" + img['file_name'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    # create masked images from annotations
    segmentation = ann['segmentation']
    segmentation = np.array(segmentation[0], dtype=np.int32).reshape(-1, 2)

    cv2.polylines(image, [segmentation], isClosed=True, color=(0, 255, 0), thickness=2)  # Green color with thickness 2

    plt.subplot(n,3,j)

    plt.imshow(image)
    j+=1
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [segmentation], color=1)
    plt.subplot(n,3,j)

    plt.imshow(mask,cmap='gray')
    j+=1

visualize_random_images()



def _train_masks():
    print('train masks')
    mask_dir = '/Users/muhammadgulfamtahir/PycharmProjects/working/train_masks/'
    os.makedirs(mask_dir, exist_ok=True)
    totalImages = len(train_annotations['images'])
    done = 0
    for img,ann in zip(train_annotations['images'],train_annotations['annotations']):
        path = train_path+ "/" + img['file_name']
        mask_path = mask_dir+ "/" + img['file_name']
        # load image in open cv
        image = cv2.imread(path)
        segmentation = ann['segmentation']
        segmentation = np.array(segmentation[0], dtype=np.int32).reshape(-1, 2)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [segmentation], color=(255,255,255))
        cv2.imwrite(mask_path, mask)
        done+=1
        print(f"train  {done} / {totalImages} ")

def _test_masks():
    print('test masks')

    totalImages = len(test_annotations['images'])
    done = 0
    mask_dir = '/Users/muhammadgulfamtahir/PycharmProjects/working/test_masks/'
    os.makedirs(mask_dir, exist_ok=True)
    
    for img,ann in zip(test_annotations['images'],test_annotations['annotations']):
        path = test_path + "/" + img['file_name']
        mask_path = mask_dir + "/" + img['file_name']
        # load image in open cv
        image = cv2.imread(path)
        segmentation = ann['segmentation']
        segmentation = np.array(segmentation[0], dtype=np.int32).reshape(-1, 2)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [segmentation], color=(255,255,255))
        cv2.imwrite(mask_path, mask)
        done+=1

        print(f"test  {done} / {totalImages} ")


def _val_masks():
    print('val masks')
    totalImages = len(val_annotations['images'])
    done = 0
    mask_dir = '/Users/muhammadgulfamtahir/PycharmProjects/working/val_masks/'
    os.makedirs(mask_dir, exist_ok=True)
    
    for img,ann in zip(val_annotations['images'],val_annotations['annotations']):
        path = val_path + "/" +  img['file_name']
        mask_path = mask_dir + "/" +  img['file_name']
        # load image in open cv
        image = cv2.imread(path)
        segmentation = ann['segmentation']
        segmentation = np.array(segmentation[0], dtype=np.int32).reshape(-1, 2)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [segmentation], color=(255,255,255))
        cv2.imwrite(mask_path, mask)
        done+=1
        print(f"val  {done} / {totalImages} ")


from threading import Thread
def make_masks():
  threads = []
  threads.append( Thread(target=_train_masks))

  threads.append( Thread(target=_test_masks))

  threads.append( Thread(target=_val_masks))
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  print('complete')
  return

make_masks()


def load_data():
    target_size = (512, 512)
    train_mask_dir = '/Users/muhammadgulfamtahir/PycharmProjects/working/train_masks/'
    X_train =  [cv2.resize(cv2.imread(train_path + "/" + image['file_name']),target_size) for image in train_annotations['images']]
    y_train = [cv2.resize(cv2.imread(train_mask_dir+ "/" +image['file_name'],cv2.IMREAD_GRAYSCALE),target_size ) for image in train_annotations['images']]
    X_train = np.array(X_train)
    y_train = np.expand_dims(np.array(y_train), axis=-1)
    
    X_train = X_train.astype('float32') / 255.0
    y_train = y_train.astype('float32') / 255.0
    y_train = (y_train > 0.5).astype(np.float32)
    
    

    

    val_mask_dir = '/Users/muhammadgulfamtahir/PycharmProjects/working/val_masks/'
    
    X_val =  [cv2.resize(cv2.imread(val_path + "/" +image['file_name']),target_size) for image in val_annotations['images']]
    y_val = [cv2.resize(cv2.imread(val_mask_dir + "/" + image['file_name'],cv2.IMREAD_GRAYSCALE),target_size) for image in val_annotations['images']]
    X_val = np.array(X_val)
    y_val = np.expand_dims(np.array(y_val), axis=-1)

    
        
    X_val = X_val.astype('float32') / 255.0
    y_val = y_val.astype('float32') / 255.0
    y_val = (y_val > 0.5).astype(np.float32)
    


    return X_train,y_train,X_val,y_val

def load_test_data():
    target_size = (512, 512)

    test_mask_dir = '/Users/muhammadgulfamtahir/PycharmProjects/working/test_masks/'
    X_test =  [cv2.resize(cv2.imread(test_path + "/" +image['file_name']),target_size) for image in test_annotations['images']]
    y_test = [cv2.resize(cv2.imread(test_mask_dir + "/" +image['file_name'],cv2.IMREAD_GRAYSCALE),target_size) for image in test_annotations['images']]
    X_test = np.array(X_test)
    y_test = np.expand_dims(np.array(y_test), axis=-1)

        
    X_test = X_test.astype('float32') / 255.0
    y_test = y_test.astype('float32') / 255.0
    y_test = (y_test > 0.5).astype(np.float32)
    return X_test,y_test


X_train,y_train,X_val,y_val = load_data()
