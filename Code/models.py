from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, HalvingGridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np


class Models_pollen():
    def __init__(self, data, botanical_species: int, seed: int, ) -> None:
        self.data=data
        self.groups=botanical_species
        self.seed=seed
    
    
    def Partition_unbalanced(self,rate=0.75):
        try:
            X1=self.data[self.data.Toma_fotografia==1].drop("Toma_fotografia",axis=1)
            X2=self.data[self.data.Toma_fotografia==2].drop("Toma_fotografia",axis=1)
            Y1=X1.Productor
            Y2=X2.Productor
            X1=X1.drop(["Productor","Nombre"],axis=1)
            X2=X2.drop(["Productor","Nombre"],axis=1)

            #Permite unir los datos de un mismo productor en uno de los grupos sin importar la toma de la fotografía
            X_train1,X_val1,Y_train1,Y_val1=train_test_split(X1,Y1,random_state=self.seed,train_size=rate)
            X_train2,X_val2,Y_train2,Y_val2=train_test_split(X2,Y2,random_state=self.seed,train_size=rate)
            #Train data
            X_train=np.vstack((X_train1,X_train2))
            Y_train=np.concatenate((Y_train1,Y_train2))
            #Validation data
            X_val=np.vstack((X_val1,X_val2))
            Y_val=np.concatenate((Y_val1,Y_val2))
            return X_train,Y_train,X_val,Y_val
        except ValueError:
            print("Check the name of the column")
        
    
    
    def Partition_balanced(self, data_control, cant_visitas: int, rate_split=2/3):

        np.random.seed(self.seed)
        X=np.array(self.data.drop(["Toma_fotografia","Productor","Nombre"],axis=1))
        Y=np.array(self.data.Productor)
        conjuntos=len(np.unique(Y))*cant_visitas*2
        #Vector de cantidades de datos por productor
        parte=np.array(data_control["Cantidad"],np.int8)
        #Vector de cantidades de datos para entrenamiento
        cant=np.int8(parte*rate_split)

        Positions=[]
        pos0=np.random.permutation(range(0,parte[0],2))
        pos1= pos0+1
        pos=[]
        pos_val=[]
        for k,j in zip(pos0,pos1):
            pos.append(k)
            pos.append(j)
        Positions.append(pos)
        train=pos[:cant[0]]
        val=pos[cant[0]:parte[0]]
        pos_val.append(val)
        X_train=X[train]
        X_val=X[val]
        Y_train=Y[train]
        Y_val=Y[val]

        for i in range(1,conjuntos):
            pos0=np.random.permutation(range(np.sum(parte[:i]),np.sum(parte[:i+1]),2))
            pos1= pos0+1
            pos=[]
            for k,j in zip(pos0,pos1):
                pos.append(k)
                pos.append(j)
                Positions.append(pos)
            train=pos[:cant[i]]
            val=pos[cant[i]:parte[i]]
            pos_val.append(val)
            X_train=np.vstack((X_train,X[train]))
            X_val=np.vstack((X_val,X[val]))
            Y_train=np.concatenate((Y_train,Y[train]))
            Y_val=np.concatenate((Y_val,Y[val]))
        pos_val=sum(pos_val,start=[])
        return X_train,Y_train,X_val,Y_val,Positions,pos_val

    
    def model_tunning(self, stimator_models: list[function] , params_dict: list[dict], names_model:list[str],
                      X_train, Y_train, score_list: list, cv=5)->list:
        
        results=[]
        for k, name in enumerate(names_model):
            search_tunning =HalvingGridSearchCV(stimator= stimator_models[k], param_grid= params_dict[k],
                                                scoring=score_list, refit=True, return_train_score=True, cv=cv)
            model = search_tunning.fit(X_train, Y_train)
            best_model = model.best_estimator_
            best_params= model.best_params_
            
            results.append((name,search_tunning,model, best_model, best_params))
        
        return results