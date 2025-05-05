from numpy.linalg import norm
from numpy import where, delete, sum, array, histogram
import cv2 as cv
from cleaning_image import Cleaning_image
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
import pickle as pick
import pandas as pd

class Data_creation():
    
    def __init__(self, botanical_species: int, seed: int) -> None:
        self.groups=botanical_species
        self.seed=seed
    
    def characteristic_vector_HSV(self, dimension_image: int, labes_colors: list, center_color_representation, cleaning_level=15):
        # dimension_image can take 2 or 3, depend on the color space you are working in
        
        proportion_vector,_=histogram(labes_colors,bins=len(center_color_representation),density=True)
        try:
            if dimension_image==3:
                rate=proportion_vector[norm(center_color_representation-array([0,0,254]),axis=1)<=cleaning_level ][0]
            elif dimension_image==2:
                rate=proportion_vector[norm(center_color_representation,axis=1)<=cleaning_level ][0]
            pos=where(proportion_vector==rate)[0][0]
            image_characteristic_vector=delete(proportion_vector,pos)
            image_characteristic_vector=image_characteristic_vector/sum(image_characteristic_vector)
            return image_characteristic_vector        
        except ValueError:
            print("Check de image dimension")
         
     
    
            
    def charac_vector_Luv(self, dimension_image: int, labes_colors: list, center_color_representation, cleaning_level=15):
        # dimension_image can take 2 or 3, depend on the color space you are working in
        
        proportion_vector,_=histogram(labes_colors,bins=len(center_color_representation),density=True)
        try:
            if dimension_image==3:
                rate=proportion_vector[norm(center_color_representation-array([255,  96, 136]),axis=1)<=cleaning_level ][0]
            elif dimension_image==2:
                rate=proportion_vector[norm(center_color_representation-array([ 96,136]),axis=1)<=cleaning_level ][0]
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
    
    
    
    def get_HSV_image(self, clean_image):
        return np.uint8(cv.cvtColor(clean_image,cv.COLOR_RGB2HSV))
    
    
    
    def get_LUV_image(self, clean_image):
        return np.uint8(cv.cvtColor(clean_image,cv.COLOR_RGB2Luv))
    
    
    
    def data_photo_collection(self, photos_list: list, color_space: str, sav:bool, sav_name: str, sav_path:str,croped_rate=0.1,
                            width_image=660, high_image=500, tr_rate = 1.05, c_rate = 1.15, disk_cleaning: int = 5):
        # Color space posibilities ["uv", "Luv", "HS", "HSV"]
        
        cleaning_image=Cleaning_image(self.groups, seed= self.seed)
        
        if color_space=="Luv":
            cleaned_photos_array=np.array([[255,  96, 136]])
            for k, photo in enumerate(photos_list):
                cleaned_image, _, _=cleaning_image.segm_polen(croped_rate=croped_rate, width_image=width_image, high_image=high_image,
                                        path_image=photo, tr_rate=tr_rate, c_rate=c_rate, disk_cleaning=disk_cleaning)
                Luv_image=self.get_LUV_image(cleaned_image).reshape(-1,3)
                cleaned_photos_array=np.concatenate([cleaned_photos_array,Luv_image])
            
        if color_space=="uv":
            cleaned_photos_array=np.array([[96, 136]])
            for k, photo in enumerate(photos_list):
                cleaned_image, _, _=cleaning_image.segm_polen(croped_rate=croped_rate, width_image=width_image, high_image=high_image,
                                        path_image=photo, tr_rate=tr_rate, c_rate=c_rate, disk_cleaning=disk_cleaning)
                uv_image=self.get_LUV_image(cleaned_image)[:,:,1:].reshape(-1,2)
                cleaned_photos_array=np.concatenate([cleaned_photos_array,uv_image])
                
        if color_space=="HSV":
            cleaned_photos_array=np.array([[0,0,254]])
            for k, photo in enumerate(photos_list):
                cleaned_image, _, _=cleaning_image.segm_polen(croped_rate=croped_rate, width_image=width_image, high_image=high_image,
                                        path_image=photo, tr_rate=tr_rate, c_rate=c_rate, disk_cleaning=disk_cleaning)
                HSV_image=self.get_HSV_image(cleaned_image).reshape(-1,3)
                cleaned_photos_array=np.concatenate([cleaned_photos_array,HSV_image])
                
        if color_space=="HS":
            cleaned_photos_array=np.array([[0,0]])
            for k, photo in enumerate(photos_list):
                cleaned_image, _, _=cleaning_image.segm_polen(croped_rate=croped_rate, width_image=width_image, high_image=high_image,
                                        path_image=photo, tr_rate=tr_rate, c_rate=c_rate, disk_cleaning=disk_cleaning)
                HS_image=self.get_HSV_image(cleaned_image)[:,:,:2].reshape(-1,2)
                cleaned_photos_array=np.concatenate([cleaned_photos_array,HS_image])
        
        if sav:
            try:
                np.savez_compressed( sav_path+sav_name+".npz",a=cleaned_photos_array[1:])
                print(f"The file was saved in .npz format in the path {sav_path}")
                return cleaned_photos_array[1:]
            except ValueError:
                print("Check the path. Try to save it by yourself or ejecut again this function.")
                return cleaned_photos_array[1:]
        else:
            return cleaned_photos_array[1:]
       
            
            
    
    def color_represent_data(self, cleaned_photos_array, sav:bool, sav_name: None, sav_path:None, data_reduction:bool, rate_data_reduction=0.5)->tuple:
        
        dimension=cleaned_photos_array.shape
        photo_quantiy=dimension[0]
        if data_reduction:
            n_clusters=int(photo_quantiy*(1-rate_data_reduction))
            print(n_clusters)
            if photo_quantiy<1000000:
                
                model_reduction=MiniBatchKMeans(n_clusters=n_clusters,max_iter=150,random_state=self.seed,batch_size=2**12).fit(cleaned_photos_array)
                data_reduced=np.array(model_reduction.cluster_centers_,dtype=np.uint8)

            
            elif photo_quantiy<10000000:
                data_reduced=np.uint8([[0]*dimension[1]])
                for i in range(5):
                    print(f"{i+1}-ésima iteración de 5")
                    new_photo_quantity=int(photo_quantiy*0.2)
                    index_group=np.random.choice(range(photo_quantiy), size=n_clusters, replace=False)
                    model_reduction=MiniBatchKMeans(n_clusters=new_photo_quantity,max_iter=100,random_state=self.seed,batch_size=2**15,).fit(cleaned_photos_array[index_group])
                    data_reduced=np.concatenate([data_reduced,np.array(model_reduction.cluster_centers_,dtype=np.uint8)])
                data_reduced=data_reduced[1:]
            
            elif photo_quantiy<25000000:
                data_reduced=np.uint8([[0]*dimension[1]])
                for i in range(10):
                    print(f"{i+1}-ésima iteración de 10")
                    new_photo_quantity=int(photo_quantiy*0.1)
                    index_group=np.random.choice(range(photo_quantiy), size=n_clusters, replace=False)
                    model_reduction=MiniBatchKMeans(n_clusters=new_photo_quantity,max_iter=75,random_state=self.seed,batch_size=2**16).fit(cleaned_photos_array[index_group])
                    data_reduced=np.concatenate([data_reduced,np.array(model_reduction.cluster_centers_,dtype=np.uint8)])
                data_reduced=data_reduced[1:]
            
            
            else:
                data_reduced=np.uint8([[0]*dimension[1]])
                for i in range(20):
                    print(f"{i+1}-ésima iteración de 20")
                    new_photo_quantity=int(photo_quantiy*0.05)
                    index_group=np.random.choice(range(photo_quantiy), size=n_clusters, replace=False)
                    model_reduction=MiniBatchKMeans(n_clusters=new_photo_quantity,max_iter=50,random_state=self.seed,batch_size=1024,).fit(cleaned_photos_array[index_group])
                    data_reduced=np.concatenate([data_reduced,np.array(model_reduction.cluster_centers_,dtype=np.uint8)])
                data_reduced=data_reduced[1:]
            
            
            model_center_representation=KMeans(n_clusters=self.groups+1,max_iter=500,random_state=self.seed).fit(data_reduced)
            centers_representation=np.uint8(model_center_representation.cluster_centers_)
        
        else:
            model_center_representation=KMeans(n_clusters=self.groups+1,max_iter=500,random_state=self.seed).fit(cleaned_photos_array)
            centers_representation=np.uint8(model_center_representation.cluster_centers_)
            
        
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
            return "Check the dimension of the columns of the characteristic data and the quantity of the samplesc per producer per visit"
        
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
