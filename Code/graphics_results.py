import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, classification_report, f1_score
from seaborn import heatmap
import cv2 as cv
from cleaning_image import Cleaning_image

class Graphic_results(Cleaning_image):
    def __init__(self, botanical_species: int, seed: int) -> None:
        super().__init__(botanical_species=botanical_species, seed=seed)


    def graphic_score_models(self, name_models, scores, Data_name ,parameters: list, taman=(10,5), lsize=12):
        score=np.array(scores)
        Data_name=np.array(Data_name)
        param=np.array(param)

        ####### Gráfica de los scores obtenidos en cada modelo para los diferentes subgrupos

        plt.figure(figsize=taman)
        for i in range(len(name_models)):
            plt.scatter(Data_name,score[:,0],color="blue", label=name_models[i],marker="s",s=100)
        plt.rc('axes', labelsize=lsize)
        plt.xticks(rotation=60, fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel("Subgrupos analizados")
        plt.ylabel("Precisión")
        plt.legend()
        plt.show()

        ######## Presentación de los mejores scores para cada modelo
        for i in range(len(name_models)):
            print("Better accuracy for model: ", np.max(score[:,i]),"Grupo obtenido",
                Data_name[np.where(score[:,i]==np.max(score[:,i]))])
            print("Parámetros: ",parameters[:,i][np.where(score[:,i]==np.max(score[:,i]))][0])
            print("________________________________________________________________________________________________________________")
    
    
    
    
    def get_results(self, y_test:list , pred_y: list, eje: list, save_image: bool, path_image: str, tittle_image="matrizconfusion",
                    xlabel='Beekeeper assigned',ylabel='Beekeeper assigned', format_image="jpg"):
        
        conf_matrix = confusion_matrix(y_test, pred_y)
        plt.figure(figsize=(7, 6))
        heatmap(conf_matrix, xticklabels=eje, yticklabels=eje, annot=True, fmt="d")
        plt.title(tittle_image)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.rc('axes', labelsize=10)
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(rotation=0,fontsize=7)
        if save_image:
            plt.savefig(path_image + "."+format_image,dpi=500,format=format_image, bbox_inches="tight")
        else:
            pass
        plt.show()
        acc=balanced_accuracy_score(y_test, pred_y)
        recall=recall_score(y_test, pred_y,average="weighted")
        preci=precision_score(y_test, pred_y,average="weighted")
        f1score=f1_score(y_test, pred_y,average="weighted")
        print(classification_report(y_test, pred_y))
        return acc, recall, preci, f1score, conf_matrix
    
    
    def color_print(self, centers_representation, botanical_species: int):
        fig,ax=plt.subplots(1,botanical_species+1)
        cont=0
        centers_representation=np.uint8(centers_representation)
        if centers_representation.shape[1]==3:
        
            for j in range(botanical_species+1):
                ax[j].set_axis_off()
                ax[j].imshow(cv.cvtColor(np.ones((4,4,3),dtype=np.uint8)*centers_representation[cont],cv.COLOR_HSV2RGB))
                cont +=1

            plt.show()
        
        elif centers_representation.shape[1]==2:
            plt.figure(figsize=(5,5))
            plt.scatter(centers_representation[:,0],centers_representation[:,1], c=range(botanical_species+1))
            plt.show()