from numpy.linalg import norm
from numpy import where, delete, sum, array, histogram
import cv2 as cv
from cleaning_image import Cleaning_image
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
import pickle as pick
import pandas as pd

class Data_creation(Cleaning_image):
    
    def __init__(self, botanical_species: int, seed: int) -> None:
        super().__init__(botanical_species=botanical_species, seed=seed)
            
            
    def charac_vector(self: int, labes_colors: list, center_color_representation: np.array, color_vector_out: np.array, cleaning_level=15):
        # dimension_image can take 2 or 3, depend on the color space you are working in
        # withe in LUV=np.array([255,  96, 136])
        # withe in HSV=np.array([0,0,254])
        
        proportion_vector,_=histogram(labes_colors,bins=len(center_color_representation),density=True)
        
        try:
            rate=proportion_vector[norm(center_color_representation-color_vector_out,axis=1)<=cleaning_level ][0] 
            pos=where(proportion_vector==rate)[0][0]
            image_characteristic_vector=delete(proportion_vector,pos)
            image_characteristic_vector=image_characteristic_vector/sum(image_characteristic_vector)
            return image_characteristic_vector        
        except ValueError:
            print("Check de image dimension")
    
   
    def columnas(self)->list:
        columnas_df=[]
        for i in range(1,self.groups+1):
            columnas_df.append("C"+str(i))
        return columnas_df
    
    
    
    def get_HSV_image(self, clean_image: np.array):
        return np.uint8(cv.cvtColor(clean_image,cv.COLOR_RGB2HSV))
    
    
    
    def get_LUV_image(self, clean_image: np.array):
        return np.uint8(cv.cvtColor(clean_image,cv.COLOR_RGB2Luv))
    
    
    
    def data_photo_collection(self, photos_list: list, color_space: str, sav:bool, sav_name: str, sav_path: str,croped_rate: float,
                            width_image:int, high_image: int, tr_rate = 1.05, c_rate = 1.15, disk_cleaning: int = 5):
        # Color space posibilities ["uv", "Luv", "HS", "HSV"]
        
        cleaning_image=Cleaning_image(botanical_species=self.groups, seed=self.seed)
        
        if color_space=="Luv":
            cleaned_photos_array=list()
            for k, photo in enumerate(photos_list):
                cleaned_image, _, _=cleaning_image.segm_polen(croped_rate=croped_rate, width_image=width_image, high_image=high_image,
                                        path_image=photo, tr_rate=tr_rate, c_rate=c_rate, disk_cleaning=disk_cleaning)
                cleaned_photos_array.append(self.get_LUV_image(cleaned_image))
            
        if color_space=="uv":
            cleaned_photos_array=list()
            for k, photo in enumerate(photos_list):
                cleaned_image, _, _=cleaning_image.segm_polen(croped_rate=croped_rate, width_image=width_image, high_image=high_image,
                                        path_image=photo, tr_rate=tr_rate, c_rate=c_rate, disk_cleaning=disk_cleaning)
                cleaned_photos_array.append(self.get_LUV_image(cleaned_image)[:,:,1:])
                
        if color_space=="HSV":
            cleaned_photos_array=list()
            for k, photo in enumerate(photos_list):
                cleaned_image, _, _=cleaning_image.segm_polen(croped_rate=croped_rate, width_image=width_image, high_image=high_image,
                                        path_image=photo, tr_rate=tr_rate, c_rate=c_rate, disk_cleaning=disk_cleaning)
                cleaned_photos_array.append(self.get_LUV_image(cleaned_image))
                
        if color_space=="HS":
            cleaned_photos_array=list()
            for k, photo in enumerate(photos_list):
                cleaned_image, _, _=cleaning_image.segm_polen(croped_rate=croped_rate, width_image=width_image, high_image=high_image,
                                        path_image=photo, tr_rate=tr_rate, c_rate=c_rate, disk_cleaning=disk_cleaning)
                cleaned_photos_array.append(self.get_LUV_image(cleaned_image)[:,:,:2])
        
        if sav:
            try:
                pick.dump(cleaned_photos_array, open(sav_path+sav_name+".pkl", 'wb'))
                print(f"The file was saved in .pkl format in the path {sav_path}")
                return cleaned_photos_array
            except ValueError:
                print("Check the path. Try to save it by yourself or ejecut again this function.")
                return cleaned_photos_array
        else:
            return cleaned_photos_array
       
            
            
    
    def color_represent_data(self, cleaned_photos_array, sub_cluster_rate:float, 
                             sub_data_rate:float, max_iter_minikmeans:int, max_iter_kmeans:int, iterations:int, 
                             sav:bool, sav_name= None, sav_path=None)->tuple:
        
        if type(cleaned_photos_array)==list:
            cleaned_photos_array=np.array(cleaned_photos_array).reshape(-1,np.array(cleaned_photos_array).shape[-1])
        
        dimension=cleaned_photos_array.shape
        photo_quantiy=dimension[0]
        if photo_quantiy<200000000:
            model_center_representation=MiniBatchKMeans(n_clusters=self.groups+1,max_iter=max_iter_minikmeans,random_state=self.seed,batch_size=2**11).fit(cleaned_photos_array)
            centers_representation=model_center_representation.cluster_centers_

        
        else:
            data_reduced=np.uint8([[0]*dimension[1]])
            for i in range(iterations):
                print(f"{i+1}-th iteration of {iterations}")
                sub_data=int(photo_quantiy*sub_data_rate)
                sub_cluster=int(photo_quantiy*sub_cluster_rate)
                index_group=np.random.choice(range(photo_quantiy), size=sub_data, replace=False)
                sub_model=MiniBatchKMeans(n_clusters=sub_cluster,max_iter=max_iter_minikmeans,random_state=self.seed,batch_size=2**11).fit(cleaned_photos_array[index_group])
                data_reduced=np.concatenate([data_reduced,np.array(sub_model.cluster_centers_,dtype=np.uint8)])
            data_reduced=data_reduced[1:]
            model_center_representation=KMeans(n_clusters=self.groups+1,max_iter=max_iter_kmeans,random_state=self.seed).fit(data_reduced)
            centers_representation=model_center_representation.cluster_centers_
            
        
        if sav:
            try:
                pick.dump(model_center_representation, open(sav_path+sav_name+".sav", 'wb'))
                print(f"The file was saved in .sav format in the path {sav_path}")
                return (model_center_representation,centers_representation)
            except ValueError:
                print("Check the path. Try to save it by yourself or ejecut again this function.")
                return (model_center_representation,centers_representation)
        else:
            return (model_center_representation, centers_representation)


    def create_df_control(self,  quantity_samples_per_producer_per_visit: list[list], producers: list, 
                                 quantity_photos_per_sample: int,quantity_visits:int ):
        list_data=list()
        for p in producers:
            for  v in range(1,quantity_visits+1):
                for q_p in range(quantity_photos_per_sample):
                      for q_p_v in quantity_samples_per_producer_per_visit:
                          for q in q_p_v:
                              for s in range(1,q+1):
                                list_data.append((p,v,q_p+1,s))
        
        return pd.DataFrame(list_data, columns=["Producer", "Visit", "Photo_number", "Number_sample"])
    
    def df_color_characteristics(self, croped_rate:float, centers_representation, model_center_representation, 
                                 cleaned_photos_color_transformed, high_image:int, width_image:int, 
                                 quantity_samples_per_producer_per_visit: list[list], producers: list, 
                                 quantity_photos_per_sample: int,quantity_visits:int, space_color:str, 
                                 sav:bool, sav_name: str, sav_path:str,):
        
        ## space_color can be ["HSV", "Luv"]
        
        df_resume=self.create_df_control(quantity_samples_per_producer_per_visit=quantity_samples_per_producer_per_visit,
                                         producers=producers, quantity_photos_per_sample=quantity_photos_per_sample,
                                         quantity_visits=quantity_visits)
        
        columns=self.columnas(self.groups)
        characteristic_data=pd.DataFrame(columns=columns)
        data_dimension=cleaned_photos_color_transformed.shape[1]
        
        if len(df_resume)!=((1-2*croped_rate)**2)*width_image*high_image or len(centers_representation)!=len(columns):
            return "Check the dimension of the columns of the characteristic data and the quantity of the samples per producer per visit"
        
        else:
            if space_color=="Luv":
                for i in range(len(df_resume)):
                    label_colors=model_center_representation.predict(cleaned_photos_color_transformed[i*((1-2*croped_rate)**2)*width_image*high_image:(i+1)*((1-2*croped_rate)**2)*width_image*high_image])
                    characteristic_data=pd.concat([characteristic_data, self.charac_vector_Luv(dimension_image=data_dimension,labes_colors=label_colors, center_color_representation=centers_representation)])
                final_data=pd.concat([df_resume, characteristic_data],axis=1)
            
            elif space_color=="HSV":
                for i in range(len(df_resume)):
                    label_colors=model_center_representation.predict(cleaned_photos_color_transformed[i*((1-2*croped_rate)**2)*width_image*high_image:(i+1)*((1-2*croped_rate)**2)*width_image*high_image])
                    characteristic_data=pd.concat([characteristic_data, self.charac_vector_Luv(dimension_image=data_dimension,labes_colors=label_colors, center_color_representation=centers_representation)])
                final_data=pd.concat([df_resume, characteristic_data],axis=1)
                
        if sav:
            try:
                final_data.to_excel(sav_path+sav_name+".xlsx", index=False, sheet_name=space_color)
                print(f"The file was saved in .xlsx format in the path {sav_path}")
                return final_data
            except ValueError:
                print("Check the path. Try to save it by yourself or ejecut again this function.")
                return final_data
        else:
            return final_data