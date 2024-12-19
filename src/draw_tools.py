import matplotlib.pyplot as plt
import numpy as np
import io

class Neuron():
    def __init__(self, x, y, activation):
        self.x = x
        self.y = y
        self.activation = activation


    def draw(self, neuron_radius):

        if self.activation > 1:
            color_intensity = 1
            red = 0
            green = 0
            blue = 1
        elif self.activation < 0:
            color_intensity = abs(self.activation)
            red = 1
            green = 0
            blue = 0
        else:
            color_intensity = self.activation
            red = 0
            green = 1
            blue = 0
        
        circle = plt.Circle(
            (self.x, self.y), 
            radius=neuron_radius, 
            edgecolor="black", 
            linewidth=1.5, 
            fill=True, 
            facecolor= (red, green, blue, color_intensity)
        )
        
        plt.gca().add_patch(circle)



class Layer():
    def __init__(self, network, number_of_neurons, activation,  number_of_neurons_in_widest_layer):
        self.vertical_distance_between_layers = 10
        self.horizontal_distance_between_neurons = 3
        self.neuron_radius = 1.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons, activation)

    def __intialise_neurons(self, number_of_neurons, activation):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y, activation[iteration])
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None
        

    def __line_between_two_neurons(self, neuron1, neuron2):
        # Determina si la línea conecta hacia arriba o hacia abajo
        if neuron2.y > neuron1.y:
            # Conexión hacia arriba
            start_x, start_y = neuron1.x, neuron1.y + self.neuron_radius
            end_x, end_y = neuron2.x, neuron2.y - self.neuron_radius
        else:
            # Conexión hacia abajo
            start_x, start_y = neuron1.x, neuron1.y - self.neuron_radius
            end_x, end_y = neuron2.x, neuron2.y + self.neuron_radius
        
        color_intensity = 0.1 #neuron1.activation  
        line_color = (0, 0, 1, color_intensity)
        
        # Dibuja la línea entre los puntos ajustados
        line = plt.Line2D((start_x, end_x), (start_y, end_y), color=line_color, linewidth=1)
        plt.gca().add_line(line)


    def draw(self, layerType=0):
        for neuron in self.neurons:
            neuron.draw( self.neuron_radius )
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons


        # if layerType == 0:
        #     plt.text(x_text, self.y, 'Input Layer', fontsize = 12)
        if layerType == -1:
            plt.text(x_text, self.y, 'Output Layer', fontsize = 12)
            for idx, neuron in enumerate(self.neurons):
                plt.text(neuron.x-self.neuron_radius/2, neuron.y + self.neuron_radius * 1.5, str(idx), fontsize=15)
        else:
            plt.text(x_text, self.y, 'Hidden Layer '+str(layerType + 1), fontsize = 12)

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons, activation):
        layer = Layer(self, number_of_neurons, activation, self.number_of_neurons_in_widest_layer)
        self.layers.append(layer)

    def draw(self):
        img_stream = io.BytesIO()
        plt.figure(figsize=(10, 8))
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw( i )

        plt.axis('scaled')
        plt.axis('off')


        plt.savefig(img_stream, format='png',  bbox_inches='tight')
        img_stream.seek(0) 
        plt.close()
        return img_stream

class DrawNN():
    def __init__( self, model, input ):
        self.input = input
        self.activations = [self.input]
        self.neural_network = []

        for i, layer in enumerate(model.layers):
            self.activations.append(layer(self.activations[-1]))
            if i > 0:
                self.neural_network.append( self.activations[-1].shape[1]) 
        
        self.activations = self.activations[-len(self.neural_network):]

    def draw( self ):
        widest_layer = max( self.neural_network )
        network = NeuralNetwork( widest_layer )

        
        for output_shape, activation in zip(self.neural_network, self.activations):
            network.add_layer(output_shape,  np.array(activation)[0])
        img_stream = network.draw()
        return img_stream