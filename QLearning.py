import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from joblib import dump, load
import os
import shutil
class QLearning:
    def __init__(self,field,path_predicts='Episodes'):
        '''
        Constructor of the class
        Parameters:
        field: field object
        path_predicts: path to save the results of the predictions'''
        #Field variables 
        self.field= field
        self.number_of_states= field.get_number_of_states()
        self.number_of_actions= field.number_of_actions
        self.q_table= np.zeros((self.number_of_states,self.number_of_actions))
        #Training variables
        self.episodes=0 #number of episodes
        self.path_predicts=path_predicts #path to save the results of the predictions
        self.total_reward=0
        self.total_reward_training=[]
        self.steps_training_average=0
        #Hyperparameters
        self.epsilon=0.1
        self.min_epsilon=0.01
        self.decay_epsilon=0.01
        self.alpha=0.1
        self.gamma=0.6

    def training(self,n_iterations=1000,max_epsilon=0.1,alpha=0.1,gamma=0.6,
                min_epsilon=0.01,decay_epsilon=0.001,save_learning=True):
        '''
        Training the model
        Parameters:
        n_iterations: number of iterations
        max_epsilon: initial epsilon value
        alpha: learning rate
        gamma: discount factor
        min_epsilon: minimum epsilon value
        decay_epsilon: decay epsilon value
        save_learning: save the q_table
        '''
        self.episodes= n_iterations
        self.epsilon=max_epsilon
        self.min_epsilon=min_epsilon
        self.decay_epsilon=decay_epsilon
        self.alpha=alpha
        self.gamma=gamma
        self.steps_training_average=0
        self.total_reward_training=[]
        steps_total=0
        for __ in range(n_iterations):
            __,steps,__=self.learning_process(copy.deepcopy(self.field),self.epsilon,
                            self.alpha,self.gamma,self.min_epsilon,self.decay_epsilon)
            self.total_reward_training.append(self.total_reward)
            steps_total+=steps
        self.steps_training_average=steps_total/(n_iterations)
        if save_learning:
            self.save_q_table(self.q_table)
        self.reset_qtable()     

    def predict(self,qtable_filename_base='q_table.joblib',hyperparams_file='params'
                ,re_training=False ,re_training_epi=1000, print_episode=False):
        '''
        Predict the model
        Parameters:
        qtable_filename_base: name of the file to save the q_table
        hyperparams_file: name of the file to save the hyperparameters
        re_training: retrain the model
        re_training_epi: number of episodes to retrain the model
        print_episode: print the episode
        '''
        if qtable_filename_base != '':
            self.q_table= self.load_q_table(qtable_filename_base)
            hyperparams= self.load_q_table(hyperparams_file)
            self.epsilon=hyperparams['epsilon']
            self.alpha=hyperparams['alpha']
            self.gamma=hyperparams['gamma']
            if re_training==False:   
                var,steps,__=self.learning_process(copy.deepcopy(self.field),self.epsilon,
                self.alpha,self.gamma,self.min_epsilon,self.decay_epsilon,print_episode)
                states_path= var.allposicions    
                self.reset_qtable() 
                return steps,self.total_reward,states_path 
            else:
                self.training(re_training_epi,self.epsilon,self.alpha,self.gamma,
                            self.min_epsilon,self.decay_epsilon,True)
                var,steps,__=self.learning_process(copy.deepcopy(self.field),self.epsilon,
                    self.alpha,self.gamma,self.min_epsilon,self.decay_epsilon,print_episode)
                states_path= var.allposicions
                self.reset_qtable() 
                return steps,self.total_reward,states_path
        else:
            return 0,0,[]

    def learning_process(self,field,epsilon=0.1,alpha=0.1,gamma=0.6,min_epsilon=0.01
                        ,decay_epsilon=0.999,print_episode=False):
        '''
        Learning process
        Parameters:
        field: field object
        epsilon: epsilon value
        alpha: learning rate
        gamma: discount factor
        min_epsilon: minimum epsilon value
        decay_epsilon: decay epsilon value
        print_episode: print the episode
        '''
        done= False
        steps=0
        self.total_reward=0
        if print_episode:
            field.graphics(field.allposicions,f"episode {steps}") 
        while not done:
            state= field.get_state()
            if random.uniform(0,1)<epsilon:
                action= random.randint(0,field.number_of_actions-1)
            else:
                action= np.argmax(self.q_table[state])

            reward,done=field.make_action(action) 
            self.total_reward+=reward 
            next_state= field.get_state()
            next_state_max= np.max(self.q_table[next_state])
            self.q_table[state,action]= (1-alpha)*self.q_table[state,action]+alpha*(reward+gamma*next_state_max- self.q_table[state,action])
            steps= steps+1
            epsilon= min_epsilon+(epsilon-min_epsilon)*np.exp(-decay_epsilon*steps)

            if print_episode==True:
                field.graphics(field.allposicions,f"episode {steps}")
                self.total_reward_training.append(self.total_reward)
                self.graphics_reward_training(f"rewars {steps}")    
        return field,steps,done 
    
    def hyperparameters_training(self,iterations,epsilon_values,alpha_values,
                                gamma_values,min_epsilon,decay_epsilon):
        '''
        Hyperparameters training
        Parameters:
        iterations: number of iterations
        epsilon_values: epsilon values
        alpha_values: alpha values
        gamma_values: gamma values
        min_epsilon: minimum epsilon value
        decay_epsilon: decay epsilon value
        '''
        best_reward = float('inf')
        best_hiperparamters = {}
        iter_=0
        for epsilon in epsilon_values:
            for alpha in alpha_values:
                for gamma in gamma_values: 
                    iter_+=1
                    # Entrena tu modelo y evalúa el rendimiento
                    self.training(iterations, epsilon, alpha, gamma,min_epsilon,decay_epsilon,False)
                    # Actualiza los mejores hiperparámetros si es necesario
                    if self.steps_training_average < best_reward:
                        best_reward = self.steps_training_average
                        best_hiperparamters = {'gamma': gamma, 'epsilon': epsilon, 'alpha': alpha}     
                    print(f'running {iter_}',' acurracy: ', best_reward)        
        print("Best hiperparameters:", best_hiperparamters)
        self.save_q_table(best_hiperparamters,'best_hiperparameters.joblib')
        return best_hiperparamters
    
    def reset_qtable(self):
        '''
        Reset the q_table
        '''
        self.q_table = np.zeros((self.number_of_states, self.number_of_actions))
    
    def save_q_table(self,q_table, filename='q_table.joblib'):
        '''
        Save the q_table
        Parameters:
        q_table: q_table
        filename: name of the file'''
        dump(q_table, filename)
        print(f'Saved to {filename}')

    def load_q_table(self,filename='q_table.joblib'):
        '''
        Load the q_table
        Parameters:
        filename: name of the file
        return: q_table
        '''
        q_table = load(filename)
        return q_table
    
    def empty_path(self,path):
        '''
        Empty the path
        Parameters:
        path: path to empty
        '''
        for nombre in os.listdir(path):
            ruta_completa = os.path.join(path, nombre)
            try:
                if os.path.isfile(ruta_completa) or os.path.islink(ruta_completa):
                    os.remove(ruta_completa)
                elif os.path.isdir(ruta_completa):
                    shutil.rmtree(ruta_completa)
            except Exception as e:
                print(f'Error {ruta_completa}. reason: {e}')

    def graphics_reward_training(self,name_fig):
        '''
        Graphics the rewards
        Parameters:
        name_fig: name of the figure
        '''
        plt.plot(self.total_reward_training)
        plt.ylabel('Values')
        plt.xlabel('Iterations')
        plt.title('Rewards')
        name_fig_path = self.path_predicts + '/' +name_fig
        plt.savefig(name_fig_path)
        plt.close()

    def analysys_qtable(self,qtable_filename_base='q_table.joblib'):
        '''
        Print the q_table
        Parameters:
        qtable_filename_base: name of the file
        '''
        if qtable_filename_base != '':
            q_table_base= self.load_q_table(qtable_filename_base)
            self.q_table=q_table_base
            list_state= []
            list_aware_max= []
            list_aware_min= []
            for i in range(self.q_table.shape[0]):
                count_aware=0
                sum_aware_max=0
                sum_aware_min=0
                for j in range(self.q_table.shape[1]): 
                    if self.q_table[i,j] != 0:
                        count_aware+=1
                    if self.q_table[i,j] > 0:
                        sum_aware_max+=self.q_table[i,j]
                    else:
                        sum_aware_min+=self.q_table[i,j]        
                list_state.append(count_aware)
                list_aware_max.append(sum_aware_max)
                list_aware_min.append(sum_aware_min)
            porcentaje_memory_usage= np.sum(list_state)/(self.q_table.shape[0]*self.q_table.shape[1])*100
            print('Porcentaje de uso de la memoria: ',porcentaje_memory_usage)
            #plt.scatter(range(len(list_state)),list_state)
            plt.hist(list_state, bins=range(0, self.q_table.shape[1]+1), alpha=0.75, rwidth=0.85)
            plt.title('Number of aware actions by state')
            plt.xlabel('States')
            plt.ylabel('Number of aware actions')
            plt.show()           
            
    def analysys_qtable_fig(self,qtable_filename_base='q_table.joblib',n_splits=50):
        '''
        Print the q_table
        Parameters:
        qtable_filename_base: name of the file
        n_splits: number of splits
        '''
        if qtable_filename_base != '':
            self.empty_path('Deep_Analysis')
            q_table_base= self.load_q_table(qtable_filename_base)
            self.q_table=q_table_base
            number_of_figs= int(self.q_table.shape[0]/n_splits)
            for fig_i in range(number_of_figs):  
                plt.figure(figsize=(10, 6))  
                data= self.q_table[fig_i*n_splits:(fig_i+1)*n_splits,:]
                masked_data = np.ma.masked_where(data == 0, data)
                cmap= plt.cm.coolwarm
                cmap.set_bad(color='black')
                norm = Normalize(vmin= np.min(data), vmax=np.max(data))
                plt.imshow(masked_data, cmap=cmap, interpolation='nearest',norm=norm)
                plt.colorbar()
                plt.title("Heatmap of Q-Table" + str(fig_i))
                plt.xlabel("Actions")
                plt.ylabel("States")
                y_labels = range(fig_i*n_splits, (fig_i+1)*n_splits)
                plt.yticks(range(n_splits), y_labels)  
                plt.tick_params(axis='y', labelsize=4)  
                name_fig_path = 'Deep_Analysis/' +'Heatmap_Qtable'+str(fig_i)
                plt.savefig(name_fig_path)
                plt.close() 

    def analysys_qtable_all_fig(self,qtable_filename_base='q_table.joblib',n_splits=50):
        '''
        Print the q_table
        Parameters:
        qtable_filename_base: name of the file
        n_splits: number of splits'''
        if qtable_filename_base != '':
            q_table_base= self.load_q_table(qtable_filename_base)
            self.q_table=q_table_base
            number_of_figs= int(self.q_table.shape[0]/n_splits)
            nrows= int(np.sqrt(number_of_figs))
            ncols= int(np.ceil(number_of_figs/nrows))
            fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*10, nrows*6))  # Ajusta el tamaño de la figura a tus necesidades
            axs = axs.flatten()
            for fig_i in range(number_of_figs):
                ax=axs[fig_i]
                im=ax.imshow(self.q_table[fig_i*n_splits:(fig_i+1)*n_splits,:], cmap='coolwarm', interpolation='nearest')
                y_labels = range(fig_i*n_splits, (fig_i+1)*n_splits)
                ax.set_yticks(range(n_splits))
                ax.set_yticklabels(y_labels)  
                ax.tick_params(axis='y', labelsize=4)  
            plt.tight_layout()
            name_fig_path = 'Deep_Analysis/' +'Heatmap_Qtable'
            plt.savefig(name_fig_path)
            plt.close()     