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
        
        
    def quantizated_color(self, image)-> list:
        image_noquantized = np.float32(image).reshape(-1,3)
        model=KMeans(n_clusters=self.groups, random_state=self.seed, tol=5e-5,max_iter=300).fit(image_noquantized)
        color_center=np.uint8(model.cluster_centers_)
        color_label=model.labels_
        inertia=model.inertia_
        image_flatten = color_center[label.flatten()]
        image_quantized= image_flatten.reshape(image_noquantized.shape)
        return image_quantized, color_label, color_center ,inertia
        
       
    
    def distance_point_line(self, pixel, director_vector=[1,1,1]):
        director_vector=array([1,1,1])
        return norm(cross(director_vector,pixel))/norm(director_vector)
    
    
    
    def centers_standar_deviation(self, centers)->float:
        normas=[]
        for center in centers:
            normas.append(self.distance_point_line(center))
        return np.std(normas)
    
    
    
    def segm_polen(self, croped_rate:float, width_image: int ,high_image: int , path_image: str,
                   tr_rate=1.05,c_rate=1.15, disk_cleaning=5):
        try:
            img=io.imread(path_image)
            a,b,_=img.shape
            if a>b:
                img=np.transpose(img,(1,0,2))
                img= resize(img, (high_image, width_image),order=1, preserve_range=True)

        except:
            a,b,_=path_image.shape
            if a>b:
                path_image=np.transpose(path_image,(1,0,2))
                img= resize(path_image, (high_image, width_image),order=1, preserve_range=True)


        i=int(croped_rate*high_image)
        ima=[]
        while i < (1-croped_rate)*high_image:
            j = int(croped_rate*width_image)
            while j < (1-croped_rate)*width_image:
                r, g, b = np.uint8(img[i, j,:3])
                ima.append([r,g,b])
                j+=1
            i+=1

        new_width_image=int((1-2*croped_rate)*width_image)
        new_high_image=int((1-2*croped_rate)*high_image)
        resized_image=np.reshape(ima,(new_high_image,new_width_image,3))

        #Computes entropy to quantifies disorder.
        #STUDY SIZE OF THE MASK
        entropy_imgage = entropy(resized_image[:,:,0], disk(disk_cleaning),)


        # Computes the threshold using Otsu method
        threshold = threshold_otsu(entropy_imgage)

        # Binarize the entropy image
        binary = entropy_imgage >= threshold*tr_rate

        # TODO: POLEN CAN OVERLAP LINES MORPHOLOGICAL OPERATORS MAY HELP OR SOME COLOR INFORMATION
        # Compute conected components in the binary image
        label_image = label(binary,background=True)
        image_label_overlay = label2rgb(label_image, image=binary)

        # Extract the contected component with the largest area
        maskRegion = np.full_like(label_image,0)
        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.area >= 100:
                maskRegion[region.coords[:,0],region.coords[:,1]] = 1

        ima_crop=np.ones((new_high_image,new_width_image,3), np.uint8)*np.array([255,255,255])
        i=0
        while i < new_high_image:
            j = 0
            while j < new_width_image:
                if maskRegion[i,j]==0:
                    ima_crop[i,j,:] = img[i, j,:]
                j+=1
            i+=1

        quantized_image,_,centers,_=self.quantizated_color(ima_crop)

        t=self.centers_standar_deviation(centers=centers)
        clean_image=np.zeros((new_high_image,new_width_image,3), np.uint8)
        for i in range(new_high_image):
            for j in range(new_width_image):
                if self.distance_point_line(quantized_image[i,j])<t*c_rate:
                    clean_image[i,j]=np.uint8([255,255,255])
                else:
                    clean_image[i,j]=ima_crop[i,j]

        return clean_image , maskRegion, ima_crop
    
