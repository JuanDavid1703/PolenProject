from numpy.linalg import norm
from numpy import array, cross
import numpy as np
from skimage.transform import resize
from skimage import io
from skimage.color import label2rgb
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.measure import label, regionprops # type: ignore
import matplotlib.patches as mpatches
from skimage.transform import resize
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans

class Cleaning_image():
    def __init__(self, botanical_species: int, seed: int) -> None:
        self.groups=botanical_species
        self.seed=seed
        
        
    def quantizated_color(self, image, colors=40)-> list:
        
        # Flatten fo the data
        image_noquantized = np.float32(image).reshape(-1,3)
        
        # Espectrum color reduction through a clustering model
        model=KMeans(n_clusters=colors, random_state=self.seed, tol=5e-5,max_iter=150).fit(image_noquantized)
        color_center=np.uint8(model.cluster_centers_)
        color_label=model.labels_
        inertia=model.inertia_
        image_flatten = color_center[color_label.flatten()]# type: ignore
        # Reduced image in the spectrum color
        image_quantized= image_flatten.reshape(image.shape)
        return image_quantized, color_label, color_center ,inertia# type: ignore
        
       
    # distance line-point
    def distance_point_line(self, pixel, director_vector=np.array([1,1,1])):
        return norm(cross(director_vector,pixel))/norm(director_vector)
    
    
    # Standar deviation of the distances between the pixels and the grayline in the RGB space
    def centers_standar_deviation(self, centers)->float:
        norms=[]
        for center in centers:
            norms.append(self.distance_point_line(center))
        return np.std(norms)# type: ignore
    
    
    # cleaning image function
    def segm_polen(self, croped_rate:float, width_image: int ,high_image: int , path_image: str, cleaning_groups= 16,
                   tr_rate=1.05,c_rate=1.15, disk_cleaning=5):
        try:
            # Use this if the path_image is a path of a file image
            img=io.imread(path_image)
            a,b,_=img.shape
            if a>b:
                # Transpose if the image is in vertical position
                img=np.transpose(img,(1,0,2))

        except:
            # Use tihs if the path_image is a numpy.array of a image
            a,b,_=path_image.shape# type: ignore
            # To standarize the sense of the photos
            if a>b:
                # Transpose if the image is in vertical position
                img=np.transpose(path_image,(1,0,2))# type: ignore


        img= resize(img, (high_image, width_image),order=1, preserve_range=True,)# type: ignore

        # Size to crop in the high of the image
        i=int(croped_rate*high_image)
        
        # Size to crop in the width of tthe image
        j=int(croped_rate*width_image)

        width_image=img.shape[1]
        high_image=img.shape[0]
        
        # Cropping the image
        resized_image=np.uint8(img[i:high_image-i,j :width_image-j])

        new_high_image=resized_image.shape[0]# type: ignore
        new_width_image=resized_image.shape[1]# type: ignore


        #Computes entropy to quantifies disorder.
            #STUDY SIZE OF THE MASK
        entropy_imgage = entropy(resized_image[:,:,0], disk(disk_cleaning),)# type: ignore


        # Computes the threshold using Otsu method
        threshold = threshold_otsu(entropy_imgage)

        # Binarize the entropy image
        binary = entropy_imgage >= threshold*tr_rate

        # TODO: POLEN CAN OVERLAP LINES MORPHOLOGICAL OPERATORS MAY HELP OR SOME COLOR INFORMATION
        # Compute conected components in the binary image
        label_image = label(binary,background=True)

        # Extract the contected component with the largest area
        maskRegion = np.full_like(label_image,0)
        for region in regionprops(label_image):
           # take regions with large enough areas
            if region.area >= 100:
                maskRegion[region.coords[:,0],region.coords[:,1]] = 1

        # White image
        ima_crop=np.ones((new_high_image,new_width_image,3), np.uint8)*np.array([255,255,255])
        i=0
        while i < new_high_image:
            j = 0
            while j < new_width_image:
                if maskRegion[i,j]==0:
                    # Assigning the original color of the pixels realted with the pollen
                    ima_crop[i,j,:] = resized_image[i, j,:]# type: ignore
                j+=1
            i+=1

        # Reduction pof the spectrum color
        quantized_image,_,centers,_=self.quantizated_color(image=ima_crop)

        # Calculate the standar deviation of the distances of the centers to the grayline
        t=self.centers_standar_deviation(centers=centers)
        clean_image=np.zeros((new_high_image,new_width_image,3), np.uint8)
        for i in range(new_high_image):
            for j in range(new_width_image):
                # Clean of the shadows and the dirt in the image through a threshold around the grayline
                # the nearest pixel to the grayline are converted in white.
                if self.distance_point_line(quantized_image[i,j])<t*c_rate:
                    clean_image[i,j]=np.uint8([255,255,255])# type: ignore
                else:
                    clean_image[i,j]=ima_crop[i,j]
        
        return clean_image , maskRegion, ima_crop
    