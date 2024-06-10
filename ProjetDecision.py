import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.stats import norm, beta




"""QUESTION 1"""
"""1. Code a python class to represent a partially defined MDP."""

class MDP:
    
    def __init__(self, initial_state,states, actions, size):
        l,h = size
        self._length = l
        self._height = min(26,h)
        self._states = states
        self._initial_state = initial_state
        self._actions = actions
        self._transitions = dict()     # format : {(state, action, next_state): probability}
        self._features = dict()        # format: {state: (φ1, φ2, φ3, φ4, φ5)} 
        self._discount_factor = 0.9
    
    def create_states(size): 
        """Cette fonction crée les états du gridworld
        
        :param (tuple) la longueur et la hauteur du gridworld
        
        Returns : la liste des états de ce gridworld
        
        """
        
        liste_states = []
        lenght, height = size
        height = min(26,height)
        for i in range(1,height+1) : 
            for number in range(1,lenght+1):
                letter = chr(64 + i)
                liste_states.append(f'{letter}{number}')
        return liste_states
    
    def _initialize_transitions(self): 
        """Fonction qui initialise toutes les transistions avec leur probabilité"""
        for state in self._states:
            for action in self._actions: 
                #Action souhaitée
                next_state = self.calculate_next_state(state, action)
                self.add_transition(state, action, next_state, 0.8)
                
                #Glissement
                slip_states = self.calculate_slip_states(state, action)
                for slip_state in slip_states:
                    self.add_transition(state, action, slip_state, 0.1)  #0.2 divisé entre les deux directions de glissement

    def add_transition(self, state, action, next_state, probability):
        """Cette fonction crée une nouvelle transition (state, action, next_state) : probability
        
        :param state l'état de depart
        :param action l'action effectuée
        :param next_state l'état atteint
        :param probability la probabilité d'observer cette transition
        
        """
        if (state, action, next_state) in self._transitions : 
            self._transitions[(state, action, next_state)] += probability
        else : 
            self._transitions[(state, action, next_state)] = probability
            
    def initialize_features(self, nb_features):
        """Initialise les états avec le vecteur 0)"""
        for state in self._states:
            self.add_feature(state,(0,)*nb_features)
            
    def add_feature(self, state, feature_names):
         """Ajoute un vecteur de caractéristiques pour un état donné."""
         self._features[state]=feature_names

    def nb_features(self): 
        return len(list(self._features.values())[0])
    
    def add_initial_state(self, state): 
        self._initial_state = state 
        
    def calculate_next_state(self, state, action):
        """ Fonction qui renvoie l'état atteint en effectuant effectuant l'action 'action' 
        
            :param state: etat actuel
            :param action: Action réalisée
            
            Returns : 
            
                state : état resultant de l'action 
            
            """
        if action == 'R':
            if int(state[1])!= self._length:
                # Si on est dans une colonne où on peut aller à droite
                next_column = int(state[1]) + 1 
                return f'{state[0]}{next_column}'
            else:
                # S'il y a un mur
                return state
        elif action == 'L':
            if int(state[1])!= 1:
                next_column = int(state[1]) - 1 
                return f'{state[0]}{next_column}'
            else:
                return state
        elif action == 'U':
            if state[0] != chr(64 + self._height):
                #Récupérer le code ASCII et convertir en caractère 
                next_row = chr(ord(state[0])+1)
                return f'{next_row}{state[1]}'
            else: 
                return state
        else:                                   
            if state[0] != 'A':
                next_row = chr(ord(state[0])-1) 
                return f'{next_row}{state[1]}'
            else: 
                return state

    def calculate_slip_states(self, state, action):
        """Fonction qui renvoie la liste des états dans lesquels on peut se trouver suite au glissement
        
            :param state: etat actuel
            :param action: Action réalisée
            
            Returns : 

                liste de state resultant du glissement
            
            """
        slip_states = []
        if action in ['L','R']:   
                   
             # Glissement vers haut ou bas
            up_state = self.calculate_next_state(state, 'U')
            down_state = self.calculate_next_state(state, 'D')
            slip_states.extend([up_state, down_state])
        else:                                               
             # Glissement vers la gauche ou la droite
             left_state = self.calculate_next_state(state, 'L')  
             right_state = self.calculate_next_state(state, 'R')
             slip_states.extend([left_state, right_state])
        return slip_states

    def give_final_next_state(self, state, action): 
        """ Fonction qui renvoie l'état s' atteint à partir de l'état state par l'action 'action' 
                    
            :param state: etat actuel
            :param action: Action réalisée
            
            Returns : 
                state : état final resultant de l'action 
            
            """
        proba = random.random()
        normal_next_states = self.calculate_next_state(state, action)
        flip_next_state = self.calculate_slip_states(state,action)
        if proba <= self._transitions[state,action,normal_next_states] : 
            return normal_next_states
        return flip_next_state[random.randint(0, 1)]
        
    def print_caracteristics(self): 
        """Fonction qui affiche toutes les caracteristiques du MDP"""
        #print("MDP ")
        #print("liste des états ", self._states)
        
        rev_list = self._states[::-1]
        for i in range(0, len(rev_list), self._length):
            print((rev_list[i:i + self._length])[::-1])   
            
        print("\n") 
             
        #print("liste des actions possibles ", self._actions)
        #print("liste des transistions ", self._transitions )
        #print(" liste features ", self._features)
        #print("discount factor ", self._discount_factor)
    
    def random_reward(nb_feature ): 
        return tuple([random.uniform(-1, 1) for i in range(nb_feature)])
    """QUESTION 2 """
    """2. Instantiate the gridworld running example."""
    
            
    def create_mdp(): 
        """ Fonction qui crée le MDP de l'enoncé 
        
            Returns : un element de la classe MDP """
        size = (8,8)
        states = MDP.create_states(size)
        actions = ['L','R','D','U']
        initial_state = 'D7'
        mdp = MDP(initial_state, states, actions, size)

        
        mdp.initialize_features(5)

        #Ajout des features
        mdp.add_feature('H3',(0,0,1,0,0))
        mdp.add_feature('H6',(0,0,1,0,0))
        mdp.add_feature('G1',(0,0,0,1,0))
        mdp.add_feature('G2',(0,0,0,1,0))
        mdp.add_feature('G3',(0,0,1,1,0))
        mdp.add_feature('G4',(0,1,0,0,0))
        mdp.add_feature('G6',(0,0,1,0,0))
        mdp.add_feature('F1',(0,0,0,1,0))
        mdp.add_feature('F2',(0,0,0,1,0))
        mdp.add_feature('F3',(0,0,0,1,0))
        mdp.add_feature('F7',(0,1,0,0,0))
        mdp.add_feature('D3',(0,0,0,0,1))
        mdp.add_feature('D4',(0,0,0,0,1))
        mdp.add_feature('D5',(0,0,0,0,1))
        mdp.add_feature('C3',(0,1,0,0,1))
        mdp.add_feature('C4',(0,0,0,0,1))
        mdp.add_feature('C5',(0,0,0,0,1))
        mdp.add_feature('B3',(0,0,1,0,0))
        mdp.add_feature('B6',(0,0,1,0,0))
        mdp.add_feature('A1',(1,0,0,0,0))
        mdp.add_feature('A3',(0,0,1,0,0))
        mdp.add_feature('A6',(0,0,1,0,0))
        
        mdp.add_initial_state('D7')

        mdp._initialize_transitions()
        
        return mdp

    """QUESTION 3 """
    """3. Provide a method to create a similar but randomly generated gridworld MDP, where the size of the grid and the number of cells with each feature can be parameterized.
    """
    def generate_random_gridworld(size, features_number):

        """
            Génère un gridworld aléatoire avec des caractéristiques données en paramètre.
            
            :param size (tuple) longueur et hauteur du gridworld.
            :param features_number (list)  avec le nombre d'états ayant cette caracteristique
            
            Returns : 
                un element de la classe MDP 
                                
            """
        
        states = MDP.create_states(size)
        actions = ['L', 'R', 'U', 'D']
        my_mdp = MDP(states[random.randint(0,len(states)-1)], states, actions, size)

        # Assignation aléatoire des caractéristiques
        my_mdp.initialize_features(len(features_number))
        my_mdp._initialize_transitions()
        
        for index in range(len(features_number)) : 
            count = features_number[index]
            if len(my_mdp._states) <  count: 
                selected_states = my_mdp._states
            else : 
                selected_states = random.sample(my_mdp._states, count)   
            for state in selected_states:
                current_features = list(my_mdp._features[state])
                current_features[index] = 1            # mise à jour de la caractéristique spécifique
                my_mdp._features[state] = tuple(current_features)

           
        return my_mdp
                    
    """QUESTION 4 """
    """4. Code a policy iteration algorithm, to solve such an MDP if a reward function is provided.
    """
    
    def random_policy(self): 
        """Génère une policy aléatoire pour chaque état 
        
            Returns : policy (dict) donnant une action pour chaque état 'state' """

        policy = {}
        liste_actions = list(self._actions)
        
        for state in self._states : 
            i =  random.randint(0, 3) # Prendre une action aléatoire
            policy[state] = liste_actions[i]
        return policy  

    def calcul_sum_with_discount_factor( self, V ,state, action) : 
        
        """ Calcule la partie de la somme avec les transistions dans ValueIteration et PolicyIteration 
        
            :param : V (dict)
            :param : state (str) un état
            :param : action (str)
            
            Return : Une partie de la somme dans value iteration
        
        """
        sum = 0 
        for s in self._states :  
            if (state,action,s) in self._transitions : 
                sum += self._transitions[state,action,s] * V[s]    
        return self._discount_factor * sum

    def verify_end(self, old_V, new_V, epsilon) : 
        """ Fonction qui verifie si la condition d'arrêt de ValueIteration est verifiée 
        
             :param old_V (dict) ancienne value pour chaque état du MDP 
             :param new_V (dict) nouvelle value pour chaque état du MDP 
             :param epsilon (float) seuil de tolérance 
             
             Returns : 
             
                True si abs(old_V - new_V) <=  epsilon """
                
        for state in self._states : 
            if abs(old_V[state] - new_V[state]) > epsilon : 
                return False 
        return True
    
    def ValueIteration (self, reward, epsilon=0.1, nb_iteration = 50): 
        #On commence avec des valeurs arbitraires ( Ici on commence à 0 )
        V = { state : 0 for state in self._states}
        new_V = dict()
        iteration = 1
        while(iteration < nb_iteration): 
            iteration += 1
            for state in self._states : 
                best_value = float('-inf')
                best_action = self._actions[0]
                for action in self._actions : 
                    value =  self.calcul_reward_function(state, reward) + self.calcul_sum_with_discount_factor(V, state, action)
                    if value > best_value : 
                        best_value = value
                        best_action = action
                new_V[state] = best_value
                    
            # Verification de la condition d'arrêt 
            if(self.verify_end(V,new_V, epsilon) == True) or iteration >= nb_iteration : 
                return new_V

    def PolicyEvaluation(self, reward_function, policy, epsilon) :  
        """Fonction qui renvoie v*(s) pour tous les états s
        
            :param reward_function (tuple) : fonction de recompense
            :param policy (dict) : action optimale pour chaque état s
            :param epsilon (float) : seuil de tolérance
            
            Returns : new_V (dict) : value pour chaque état"""
        
        #On commence avec des valeurs arbitraires ( Ici on commence à 0 )
        V = { state : 0 for state in self._states}
        new_V = dict()
        
        for state in self._states : 
            new_V[state] = self.calcul_reward_function(state, reward_function) + self.calcul_sum_with_discount_factor(V, state, policy[state])
        
        # Verification de la condition d'arrêt 
        while(self.verify_end(V,new_V, epsilon) == False ): 
            V = new_V.copy()
            for state in self._states : 
                new_V[state] = self.calcul_reward_function(state, reward_function) + self.calcul_sum_with_discount_factor( V, state, policy[state])
          
        return new_V
        
    def calcul_reward_function(self, state, reward_function) : 
        """Fonction qui renvoie R(s) pour un état donné 
        
            :param state (str) : un état
            :param reward_function (tuple) : une fonction de récompense 
            
            Returns : R(s) (float) """
        sum = 0 
        feature = self._features[state]

        for i in range(len(feature)): 
            sum+= feature[i] * reward_function[i]
        return sum
    
    def find_best_action(self,best_value_function, state, reward_function ) :
        """ Fonction qui renvoie la liste des meilleurs actions à effectuer à partir de l'état state 
        
            :param best_value_function (dict) : V*(s) pour chaque état s
            :param state (str) : un état
            :param reward_function (tuple) : fonction de récompense 
            
            Returns : liste_new_best_action (list) : liste des meilleurs actions
            """
        liste_action = list(self._actions) 
        best_value = -float('inf')
        liste_new_best_action = []
        
        #On parcourt toutes les actions possibles 
        
        for action in liste_action : 
            value = self.calcul_reward_function( state, reward_function) + self.calcul_sum_with_discount_factor(best_value_function,state, action)
            if value > best_value : 
                liste_new_best_action = []
                liste_new_best_action.append(action)
                best_value = value
            elif value == best_value : 
                liste_new_best_action.append(action)
        return liste_new_best_action

    def PolicyIteration(self,reward_function,actual_best_policy): 
        """ Fonction qui calcule la policy optimale
        
            :param reward_function (tuple) : fonction de récompense
            :param actual_best_policy (dict) : policy actuelle
            
            Returns actual_policy, best_value_function (tuple) : nouvelle meilleure policy , nouvelle value 

        """
        
        #Si aucune policy n'a été choisie pour le moment
        if actual_best_policy == None : 
            actual_policy = self.random_policy()
        else : 
            actual_policy  = actual_best_policy
            
        best_value_function = self.PolicyEvaluation(reward_function, actual_policy, 10**(-3))
        new_policy = {}
        finish = False 
        
        while (finish == False ):
            finish = True
            for state in  self._states : 
                set_best_actions = self.find_best_action(best_value_function,state,reward_function ) #On cherche les meilleurs actions pour chaque état
                if actual_policy[state] in set_best_actions : # Si la liste des meilleurs actions contient l'ancienne meilleure action de l'ancienne policy, on la garde
                    new_policy[state] = actual_policy[state]
                else :  # Sinon on change d'action pour l'état state. 
                    finish = False # On a réalisé un changement donc on ne se situe par encore dans un état stationnaire. 
                    new_policy[state] = list(set_best_actions)[0] # On prend une des meilleurs actions
            if finish == True : # Si aucun changement d'action n'a été fait, alors la nouvelle policy est égale à l'ancienne et l'algorithme s'arrête
                return actual_policy, best_value_function
            else : # On actualise la policy et on recommence la boucle 
                actual_policy = new_policy
                
                best_value_function = self.PolicyEvaluation(reward_function, actual_policy,0.1)
                
        return actual_policy, best_value_function
       
    def print_best_policy(self, best_policy): 
        """Cette fonction affiche les directions optimales pour chaque états d'après la policy optimale 
        
        :param best_policy (dict) la policy optimale pour chaque état"""
        directions = {'U' : '↑','D': '↓','L': '←','R': '→'}
        liste_directions = [directions[best_policy[state]] for state in self._states ]
        liste_directions = liste_directions[::-1]
        for i in range(0, len(self._states), self._length):
                print((liste_directions[i:i + self._length])[::-1]) 
    """QUESTION 5 """
    """ Provide a method which, given a reward function R, a number M of timesteps, and an initial state s0, provides a set O of M state-action pairs starting from s0. 
    To simulate an imperfect tutor, the optimal action will be chosen with probability 0.95, and a random action is chosen otherwise.
    """
    
    def M_state_actions(self, M ,reward_function, s0): 
        
        """ Fonction qui calcule un ensemble de M (state, actions) utilisant la PolicyIteration 
        
            :param M (int) : nombre d'itérations voulu
            :param reward_function (tuple) : fonction de récompense
            :param s0 (str) : état initial
            
            Returns : liste de state,action
            
            """
    
        states_actions = []
        best_policy, _ = self.PolicyIteration(reward_function, None) 
        actual_state = s0 # Etat initial 
        
        for i in range(M): 
            nb = random.random()
            
            if nb <= 0.95 : 
                #Choix de la meilleure action 
                action = best_policy[actual_state]
                states_actions.append((actual_state, action))
                
            else : 
                #Choix aléatoire
                i = random.randint(0,len(self._actions)-1)
                action = ['L', 'R', 'U', 'D'][i]
                states_actions.append((actual_state, action))
                
            # Calcul du nouvel état
            actual_state = self.give_final_next_state(actual_state,action )
        return states_actions
    
    """QUESTION 6 """
    """Provide a method to compute the ratio P(R1|O)/P(R2|O) given R1, R2, O, and a choice for the prior function. 
    This is a meta step which will require to provide several classes and methods."""
    
    def calculate_Q(self, action, state, best_value, reward_function): 
        
        """"
        Fonction qui calcule q∗(s, a), the quality of performing action a in state s 
        
            :param action (str) 
            :param state (str)
            :param best_value (dict)
            :param reward_function (tuple) : fonction de récompense
            
            Returns : Q(s,a) """
        sum = 0 
        for s1,a,s2 in self._transitions : 
            if s1 == state and a == action : 
                sum +=  self._transitions[s1,a,s2] * best_value[s2]
        return self.calcul_reward_function(state,reward_function ) + self._discount_factor*sum
    
    
class BayesianFramework:
    
    def __init__(self, prior_distribution,param_prior, mdp, alpha, M_state_action ):
        self._prior = prior_distribution
        self._param_prior = param_prior
        self._mdp = mdp
        self._alpha = alpha
        self._M_state_action = M_state_action
       
    def print_caracteristics_Bayesian_networks(self) : 
        """ Fonction qui affiche les caractéristiques du Bayesian Networks """
        print("Distribution de probabilité ", self._prior)
        self._mdp.print_caracteristics()
        print("alpha ", self._alpha)

    def calculate_likelihood(self ,reward_function, states_actions) :
        
        """ Calcul de P(O|R) 
        
        :param best_policy (dict) : meilleure policy actuelle 
        :param best_value (dict) : meilleure value 
        :param reward_function (tuple) : fonction de récompense
        :param states_actions (list) : liste de states-actions
        
        Returns : probabilité P(O|R) """
        best_value = self._mdp.ValueIteration(reward_function )
        product = 1
        for si,ai in states_actions:
            numerateur = math.exp(self._alpha*self._mdp.calculate_Q(ai, si,best_value,  reward_function))
            sum = 0
            for action in self._mdp._actions :
                sum += math.exp(self._alpha * self._mdp.calculate_Q(action, si, best_value, reward_function))
            product*= numerateur/ sum
        return product

    def Gaussian_prior(self, reward_function): 
        """Fonction qui calcule P(R) sachant que la distribution de probabilité est gaussienne 
        
        :param  reward_function (tuple) : une fonction de récompense
        :param variance (float) 
        
        Return P(R) """
        variance = self._param_prior
        return np.prod(norm.pdf(reward_function, 0, variance))
    
    def Beta_prior(self, reward_function) : 
        """Fonction qui calcule P(R) sachant que la distribution de probabilité est Beta
        
        :param  reward_function (tuple) : une fonction de récompense
        
        Return P(R) 
        
        """
        Rmax = self._param_prior
        res = 1 
        for ri in list(reward_function):
            if -Rmax <= ri <= Rmax: 
                r_normalized = (ri+Rmax)/(2*Rmax)
                res *= beta.pdf(r_normalized, 1/2, 1/2)
            else : 
                return 0
        return res
        
    def Uniform_prior(self, reward_function): 
        Rmax = self._param_prior
        if all(-Rmax <= r <= Rmax for r in reward_function) : 
            return 1/(2*Rmax)**len(reward_function)   
        return 0
    
    def Probability_R(self, reward_function): 
        """ Fonction qui calcule P(R) en fonction de la distribution de probabilité choisie 
        
        :param  reward_function (tuple) : une fonction de récompense
                
        Return P(R) """
        
        if self._prior == "Uniform" :
            return self.Uniform_prior(reward_function)
            
        elif self._prior == "Beta" : 
            return self.Beta_prior(reward_function)
                    
        return self.Gaussian_prior(reward_function)
        
    def calcul_ratio(self, reward_function_1, reward_function_2, O):
        """ Fonction qui calcule le ration P(R1|O)/P(R2|O) 
        
        :param reward_function_1 (tuple) : fonction de recompense 1
        :param reward_function_2 (tuple) : fonction de recompense 2
        :param O (list) : liste de state-action
        
        Returns : P(R1|O)/P(R2|O) """

        conditional_proba1 = self.calculate_likelihood( reward_function_1, O)
        prior_r1 = self.Probability_R(reward_function_1)
        
        conditional_proba2 = self.calculate_likelihood( reward_function_2, O)
        prior_r2 = self.Probability_R(reward_function_2)
        if prior_r1*prior_r1*conditional_proba1*conditional_proba2 == 0 : 
            return 0
        
        return (conditional_proba1 * prior_r1) / (conditional_proba2 * prior_r2)
            
    """QUESTION 7"""
    
    """Code the PolicyWalk algorithm to return a reward with high a posteriori probability. """
    def probability_distribution_R(self, R_max, l):
        """Fonction qui discretise les reward function avec le paramètre l
        
        :param R_max (float) 
        :param l (float) : intervalle entre les valeurs
        
        Returns : reward_value (list) : liste des valeurs possibles entre -R_max et R_max"""

        return list(np.arange(-R_max, R_max + l , l))

    def random_reward_function(self, nb_features, reward_value):
        """Fonction qui crée une reward_function aléatoirement 
        :param nb_features (int) 
        :param reward_value (list) : liste de differentes valeurs possibles 
        
        Returns : reward_function (list) : une fonction de récompense"""
        n = len(reward_value)
        reward_function = [ reward_value[random.randint(0,n-1)] for i in range(nb_features)]
        
        return reward_function

    def list_of_neighbours(self,reward_function, l):
        """Fonction qui crée la liste de tous les neighbors de la reward function
        
        :param reward_function (tuple) : une fonction de récompense
        :param l (float) : seuil de tolérance
        
        Returns : une liste de fonctions de recompense voisines de reward_function """
        
        list = []
        n = len(reward_function)
        for i in range(n):
            copy_reward = reward_function.copy()
            copy_reward[i]+= l
            list.append(copy_reward)
            copy_reward[i]-= 2*l # -l pour revenir a la reward normale et -l pour l'écart de l
            list.append(copy_reward)
        return list

    def find_other_policy(self, best_policy,best_value, neighbor) : 
        """  Fonction qui cherche si il existe (s,a) telle que Qπ(s,π(s),R′)<Qπ(s,a,R′)
        
        :param best_policy (dict) : la meilleure policy actuelle
        :param best_value (dict) : la meilleure value actuelle
        :param neighbor (tuple) : une fonction de recompense
        
        Returns : True si il existe (s,a) telle que Qπ(s,π(s),R′)<Qπ(s,a,R′)
        
        """
        
        M = self._mdp
        liste_states = list(M._states)
        liste_actions = list(M._actions)
        for state in liste_states :
            for action in liste_actions : 
                if M.calculate_Q( action,state, best_value, neighbor) > M.calculate_Q( best_policy[state],state,best_value, neighbor) : 
                    return True 
        return False
             
    def PolicyWalk(self,l, nb_iterations, M_state_action, Rmax) :
        
        """Fonction qui renvoie une fonction de récompense 
        :param l (float) 
        :param nb_iterations (int) nombre d'iterations 
        :param M_state_action (list) une observation de M state-action
        
        Return : une fonction de récompense (tuple) """
        
        M = self._mdp
        
        reward_value =  self.probability_distribution_R(Rmax, l) #Liste de toutes les valeurs que peut prendre la reward function 
        nb_features = M.nb_features() 

        choice_reward = self.random_reward_function(nb_features, reward_value) #Choix aléatoire d'une reward function 
        best_policy, best_value = M.PolicyIteration(choice_reward, None)
        iteration = 1
        
        while (iteration <= nb_iterations) :
            iteration+= 1
            list_neighbours = self.list_of_neighbours(choice_reward, l)
            neighbour = list_neighbours[random.randint(0,len(list_neighbours)-1)] #choix d'une reward function uniformement dans la liste des neighbours
            
            if self.find_other_policy(best_policy,best_value, neighbour) == True : 
                policy_prime, best_value_prime =  M.PolicyIteration(neighbour,best_policy )
                
                mini = min(1, self.calcul_ratio( neighbour, choice_reward, M_state_action) )
                proba = random.uniform(0,1)
                if proba <= mini :
                    choice_reward = neighbour
                    best_policy = policy_prime
                    
            else : 
                mini = min(1, self.calcul_ratio( neighbour, choice_reward, M_state_action) )

                proba = random.uniform(0,1)
                if proba <= mini :
                    choice_reward = neighbour
        
        return choice_reward, best_policy

    """QUESTION 8 """
    """Provide such a modified version of your PolicyWalk algorithm."""
    
    def calcul_reffroiddissement(self, T_initial, taux_de_reffroidissement, iteration):
        """Fonction qui update la température T pour une itération donnée
        
        :param initial_temp_initiale (float) : température initiale
        :param taux_de_reffroidissement (float) 
        :param iteration (int)
        
        Return : une nouvelle température (float) """
        
        return T_initial * (taux_de_reffroidissement ** iteration)
    
    def calculate_1Ti(self, i):
        """Fonction qui renvoie 1/Ti à la i-eme itération de l'algorithme PolicyWalkModified en fonction des données des articles 
        :param i l'iteration de l'algorithme
        
        Return 1/Ti""" 
        
        return 25+i/50
        
    def PolicyWalkModified(self,l, nb_iterations,T_initial,taux_reffroiddissement ,M_state_action, Rmax) :
        
        """Fonction qui renvoie une fonction de récompense 
        :param l (float) 
        :param nb_iterations (int) nombre d'iterations 
        :param M_state_action (list) une observation de M state-action
        
        Return : une fonction de récompense (tuple) """
        
        M = self._mdp
        
        reward_value =  self.probability_distribution_R(Rmax, l) #Liste de toutes les valeurs que peut prendre la reward function 
        nb_features = M.nb_features() 

        choice_reward = self.random_reward_function(nb_features, reward_value) #Choix aléatoire d'une reward function 
        best_policy, best_value = M.PolicyIteration(choice_reward, None)
        iteration = 1
        Ti = T_initial
        
        while (iteration <= nb_iterations) :
            iteration+= 1
            list_neighbours = self.list_of_neighbours(choice_reward, l)
            neighbour = list_neighbours[random.randint(0,len(list_neighbours)-1)] #choix d'une reward function uniformement dans la liste des neighbours
            Ti = Ti*taux_reffroiddissement
            mini = min(1, self.calcul_ratio( neighbour, choice_reward, M_state_action)**(1/Ti) )
            if self.find_other_policy(best_policy,best_value, neighbour) == True : 
                policy_prime, best_value_prime =  M.PolicyIteration(neighbour,best_policy )
                proba = random.uniform(0,1)
                if proba <= mini :
                    choice_reward = neighbour
                    best_policy = policy_prime
                    
            else : 
                proba = random.uniform(0,1)
                if proba <= mini :
                    choice_reward = neighbour
        
        return choice_reward, best_policy

    
    """QUESTION 9"""
    """9 Perform experiments on gridworld-like MDPs of varying characteristics. 
    You will obtain a variety of plots giving the performance of your algorithms 
    with different choices of parameters as a function of some other parameters 
    (e.g., size of the MDP, number of features, number of iterations in the PolicyWalk algorithm, ...)."""
       
    def calcul_loss(self, best_policy, state_actions): 
        """Fonction qui calcule le nombre d'erreurs faites dans state_actions
        :param best_policy (dict) : meilleure policy actuelle
        :param state_actions (list) : liste de state-actions
        
        Return : (int) un nombre d'erreurs réalisées """
        return len([state for state, action in state_actions if action != best_policy[state]])
         
    def mean_loss(self,state_actions,l, iteration, Rmax ): 
        """Fonction qui renvoie la moyenne de la loss_function obtenue à partir de 10 exécutions de PolicyWalk
        :param state_actions : Les observations 
        :param l : le stepsize
        :param iteration : le nombre d'iteration de l'algorithme
        :param Rmax : le max de la reward function
        
        Returns la moyenne (float) des loss des 10 executions de l'algorithme PolicyWalk"""
        liste_loss = []
        for i in range(10): 
            _, actual_policy = self.PolicyWalk(l, iteration, state_actions, Rmax)
            liste_loss.append(self.calcul_loss(actual_policy,state_actions ))
        return sum(liste_loss) / len(liste_loss)
    

def random_feature(n, nb_features):
    """Renvoie un tuple de 'nb_features' caractéristiques avec le nombre de cases ayant cette caractéristique dans un gridworld de n cases
    :param n (int) le nombre de cases dans le gridworld
    :param nb_features (int) le nombre de caracteristiques du gridworld
    
    Returns : un tuple où chaque élément représente une caractéristique et son nombre de cases
    
    """
    
    repartition = [random.randint(0, n//3) for _ in range(nb_features)]
    somme_repartition = sum(repartition)
    if somme_repartition == 0 :
        repartition[0] = 1
        somme_repartition+=1
        
    repartition = [int(x * n / somme_repartition) for x in repartition]
    difference = n - sum(repartition)
    for i in range(difference):
        repartition[i] += 1
    return tuple(repartition)

def generate_N_random_gridword(): 
    """Cette fonction génère N random gridworld """
    number_features = 5 
    liste_gridword = []
    for lenght in range (1,50): 
        for height in range(1,50): 
            f = max(lenght*height, lenght*height-15)
            feature = list(random_feature(f,number_features))
            liste_gridword.append(MDP.generate_random_gridworld((lenght, height), feature)) 
    return liste_gridword

def plot_perf(parameter,parameter_name, loss): 
    plt.scatter(parameter, loss)
    plt.xlabel("stepsize")
    plt.ylabel('Loss')
    plt.title("Evolution de la loss function en fonction " + parameter_name)
    plt.show()

def perf_from_size(iteration): 
    """Renvoie la le score de perte en fonction de la taille du gridworld"""
    reward_function = (1,2,5,0,-2)
    Rmax = max([abs(i) for i in list(reward_function)])

    l = Rmax/10
    lenghts = [i for i in range(2,21)]
    loss = []
    for lenght in lenghts : 
        my_mdp = MDP.generate_random_gridworld((lenght, lenght), [1,lenght-1, 2, 1, lenght])
        M_states_actions =  my_mdp.M_state_actions(10 ,reward_function, my_mdp._initial_state)  
        normalized = len(M_states_actions)  
        by = BayesianFramework("Uniform", max(reward_function),my_mdp, 0.9,M_states_actions )
        _, actual_policy = by.PolicyWalk(l, iteration, M_states_actions, Rmax)
        loss.append(by.calcul_loss(actual_policy,M_states_actions)/normalized)
    return lenghts, loss

def perf_from_iteration(): 
    """Renvoie la le score de perte en fonction du nombre d'itération de l'algorithme PolicyWalk"""

    liste_iteration  = list(range(50, 500, 20))

    loss = []
    my_mdp = MDP.generate_random_gridworld((14, 14), [1,12, 2, 1, 5])
    reward_function = (1,2,5,0,-2)
    Rmax = max([abs(i) for i in list(reward_function)])
    l = Rmax/5 
    M_state_action = my_mdp.M_state_actions(10,reward_function, my_mdp._initial_state)
    normalized = len(M_state_action)
    by = BayesianFramework("Uniform",Rmax, my_mdp, 0.95, M_state_action )
    for iteration in liste_iteration : 
        _ , policy = by.PolicyWalk(l,iteration, by._M_state_action, Rmax)
        perte = by.calcul_loss(policy,M_state_action ) /normalized
        loss.append(perte)
        print(iteration, perte)
    return liste_iteration, loss

def perf_from_number_features(): 
    """Renvoie le score de perte en fonction du nombre de caracteristiques du gridworld de l'algorithme PolicyWalk"""
    liste_features  = list(range(2, 10))
    loss = []
    for nb_features in liste_features : 
        my_mdp = MDP.generate_random_gridworld((10,10), random_feature(10*10,nb_features ))
        reward_function = MDP.random_reward(nb_features)
        Rmax = max([abs(i) for i in list(reward_function)])
        l = Rmax/3 
        M_state_action = my_mdp.M_state_actions(10,reward_function, my_mdp._initial_state)
        normalized = len(M_state_action)
        by = BayesianFramework("Uniform",Rmax, my_mdp, 0.95,M_state_action )
        _ , policy = by.PolicyWalk(l,300, by._M_state_action, Rmax)
        perte = by.calcul_loss(policy,M_state_action ) /normalized
        loss.append(perte)
        print(nb_features, loss)
    return liste_features, loss

def perf_from_stepsize(): 
    """Renvoie le score de perte en fonction du stepsize utilisé lors de l'algorithme de PolicyWalk"""
    reward_function = (1,2,5,0,-2)
    Rmax = max([abs(i) for i in list(reward_function)])
    liste =  list(range(2, 70, 10))
    stepsizes = [Rmax/i for i in liste]
    my_mdp = MDP.create_mdp()
    M_states_actions =  my_mdp.M_state_actions(10 ,reward_function, my_mdp._initial_state)   
    normalized = len(M_states_actions)
    by = BayesianFramework("Uniform",Rmax,my_mdp, 0.9,M_states_actions )
    loss = []
    for stepsize in stepsizes : 
        _, actual_policy = by.PolicyWalk(stepsize, 300, M_states_actions, Rmax)
        perte = by.calcul_loss(actual_policy,M_states_actions )/normalized
        print(stepsize, perte)
        loss.append(perte)
    return stepsizes, loss
    
reward_function  = (1,2,5,0,-2)
alpha = 0.95

#Rmax = max(reward_function)
#nb_features, loss = perf_from_number_features()

#taille du gridworld
#[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] , [1, 1, 0.6, 0.9, 0.6, 0.6, 1, 0.4, 0.4, 0.2, 0.2, 0.4, 0.4, 0.8, 0.5, 0.6, 0.1, 0.2, 0.4]
 
#nombre d'itérations 
#[50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350, 370, 390, 410, 430, 450, 470, 490], [ 
# [0.9, 1.0, 1.0, 0.8, 0.9, 0.8, 0.0, 1.0, 0.3, 0.0, 0.7, 0.0, 0.4, 0.2, 0.5, 0.4, 0.1, 0.1, 0.0, 0.3, 0.4, 0.5, 0.3]

#nombre de features
#[2, 3, 4, 5, 6, 7, 8, 9]
#[1.0, 0.4, 0.6, 0.8, 1.0, 0.9, 0.0, 0.9]

#stepsize
#[2, 12, 22, 32, 42, 52, 62]
#[0.7, 0.1, 0.7, 0.9, 0.5, 0.4, 0.5]
