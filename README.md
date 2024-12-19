# InteractiveMNISTPPreds

## Overview

**InteractiveMNISTPPreds** is a fun and educational project designed to explore and visualize the inner workings of a neural network trained on the MNIST dataset. While the accuracy or complexity of the model isn't the primary focus, the project emphasizes interactive visualization of the neural network's activations as it processes input data.

The graphical interface displays two key sections:
- **Input Section**: Allows users to draw or provide an input digit from the MNIST dataset.
- **Network Visualization**: Shows the activations of neurons across the network layers in real-time, with colors indicating the intensity of activation and lines representing the connections between neurons.

This project provides an engaging way to understand how a neural network processes information and propagates activations through its layers.

![Interactive Visualization Example](./img/sample.gif)

---

## Features

- **Real-Time Visualizations**: Observe how input digits activate neurons across layers.  
- **Interactive Interface**: Draw input digits and see immediate network responses.  
- **Customizable Model**: Swap in any pre-trained model to explore its activations.  
- **Educational Insight**: Gain a deeper understanding of neural network mechanics.

---

## Installation

Follow these steps to set up the project on your local machine:

1. Create a virtual Python environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the environment:
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Navigate to the `src` directory:
   ```bash
   cd src
   ```

2. Run the application:
   ```bash
   python main.py
   ```

Once the application starts, interact with the interface to draw input digits and observe how the network processes them in real-time.

---

## How It Works

1. **Neural Network**: A simple pre-trained model with two hidden layers and one output layer is used for prediction.  
2. **Visualization**: Neurons are drawn as circles, with their activation intensity represented by color:
   - **Green**: Positive activations.  
   - **Red**: Negative activations.  
   - **Blue**: Maximum intensity.  
3. **Connections**: Lines between neurons represent weights, with color intensity reflecting the connection's strength.  
4. **Interactive Interface**: Built using Tkinter, allowing users to draw inputs and view activations dynamically.

---

## Contributing

Contributions are welcome! If you'd like to improve this project, follow these steps:  
1. Fork the repository.  
2. Create a feature branch:  
   ```bash
   git checkout -b feature-name
   ```  
3. Commit your changes and push:  
   ```bash
   git push origin feature-name
   ```  
4. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

---

## Acknowledgments

Special thanks to:
- TensorFlow and Keras for providing powerful deep learning tools.
- The MNIST dataset for being a classic benchmark in the machine learning community.
- The creators of Python and Tkinter for making interactive GUI development accessible.