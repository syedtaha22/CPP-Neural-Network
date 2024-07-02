#include <iostream>
#include <vector>
#include <cmath>
#include <fstream> // For file operations

// ReLU activation function: Returns max(0, x) where x is the input value.
double relu(const double &x) { return x > 0 ? x : 0; }

// Derivative of ReLU: Returns 1 if input > 0, else 0.
double relu_derivative(const double &x) { return x > 0 ? 1 : 0; }

// Linear activation function: Returns the input value (identity function).
double linear(const double &x) { return x; }

// Derivative of Linear activation function: Always returns 1.0.
constexpr double linear_derivative(const double &x) { return 1.0; }

// Dense Layer Class template for a fully connected layer
template <int InputSize, int OutputSize>
class Dense_Layer
{
public:
    std::vector<double> inputs;               // Input values to the layer
    std::vector<double> outputs;              // Output values from the layer
    std::vector<std::vector<double>> weights; // Weight matrix
    std::vector<double> biases;               // Bias vector
    bool use_relu;                            // Flag to use ReLU activation

    // Constructor initializes layer with optional ReLU activation
    explicit Dense_Layer(const bool relu = true) : use_relu(relu)
    {
        outputs.resize(OutputSize);
        weights.resize(InputSize, std::vector<double>(OutputSize));
        biases.resize(OutputSize);
        init_weights_biases();
    }

    // Initialize weights and biases with random values
    void init_weights_biases()
    {
        srand((unsigned)time(0)); // Seed random number generator
        for (int i = 0; i < InputSize; ++i)
            for (int j = 0; j < OutputSize; ++j)
                weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1; // Initialize with values in range [-1, 1]
        for (int j = 0; j < OutputSize; ++j)
            biases[j] = ((double)rand() / RAND_MAX) * 2 - 1; // Initialize biases similar to weights
    }

    // Forward pass through the layer --> A(∑x.w + b)
    void forward(const std::vector<double> &input)
    {
        inputs = input;
        for (int j = 0; j < OutputSize; ++j)
        {
            outputs[j] = biases[j];
            for (int i = 0; i < InputSize; ++i)
            {
                outputs[j] += inputs[i] * weights[i][j];
            }
            // Apply activation function (ReLU or linear)
            outputs[j] = use_relu ? relu(outputs[j]) : linear(outputs[j]);
        }
    }

    // Backward pass through the layer to update weights and biases
    void backward(const std::vector<double> &errors, std::vector<double> &input_errors, double learning_rate)
    {
        std::vector<double> gradients(OutputSize);
        // Compute gradients based on activation function's derivative
        for (int j = 0; j < OutputSize; ++j)
        {
            gradients[j] = errors[j] * (use_relu ? relu_derivative(outputs[j]) : linear_derivative(outputs[j]));
        }

        // Update weights and compute input errors
        for (int i = 0; i < InputSize; ++i)
        {
            input_errors[i] = 0.0;
            for (int j = 0; j < OutputSize; ++j)
            {
                input_errors[i] += gradients[j] * weights[i][j];
                weights[i][j] += learning_rate * gradients[j] * inputs[i];
            }
        }

        // Update biases
        for (int j = 0; j < OutputSize; ++j)
        {
            biases[j] += learning_rate * gradients[j];
        }
    }
};

// Model Class manages the network structure and training
class Model
{
public:
    Dense_Layer<1, 10> layer1;       // First hidden layer
    Dense_Layer<10, 10> layer2;      // Second hidden layer
    Dense_Layer<10, 1> output_layer; // Output layer

    // Constructor initializes layers
    Model() : layer1(true), layer2(true), output_layer(false) {} // ReLU for hidden layers, linear for output layer

    // Forward pass through the network
    void forward(const std::vector<double> &input)
    {
        layer1.forward(input);
        layer2.forward(layer1.outputs);
        output_layer.forward(layer2.outputs);
    }

    // Backward pass to update weights and biases
    void backward(const std::vector<double> &target, double learning_rate)
    {
        std::vector<double> output_errors(1);
        output_errors[0] = target[0] - output_layer.outputs[0];

        std::vector<double> layer2_errors(10);
        output_layer.backward(output_errors, layer2_errors, learning_rate);

        std::vector<double> layer1_errors(10);
        layer2.backward(layer2_errors, layer1_errors, learning_rate);

        std::vector<double> input_errors(1);
        layer1.backward(layer1_errors, input_errors, learning_rate);
    }

    // Compute Mean Squared Error loss
    double compute_loss(const std::vector<double> &target) const
    {
        const double error = target[0] - output_layer.outputs[0];
        return 0.5 * error * error;
    }

    // Train the neural network
    void train(const std::vector<std::vector<double>> &training_inputs, const std::vector<std::vector<double>> &training_outputs, int epochs, double learning_rate)
    {
        for (int e = 0; e < epochs; ++e)
        {
            double total_loss = 0.0;
            for (int i = 0; i < training_inputs.size(); ++i)
            {
                forward(training_inputs[i]);
                total_loss += compute_loss(training_outputs[i]);
                backward(training_outputs[i], learning_rate);
            }
            std::cout << "Epoch " << e + 1 << ", Loss: " << total_loss / static_cast<double>(training_inputs.size()) << std::endl;
        }
    }

    // Predict output for a given input
    std::vector<double> predict(const std::vector<double> &input)
    {
        forward(input);
        return output_layer.outputs;
    }

    // Evaluate model performance on test data
    void evaluate(const std::vector<std::vector<double>> &test_inputs, const std::vector<std::vector<double>> &test_outputs)
    {
        double total_loss = 0.0;
        for (int i = 0; i < test_inputs.size(); ++i)
        {
            forward(test_inputs[i]);
            total_loss += compute_loss(test_outputs[i]);
        }
        std::cout << "Evaluation Loss: " << total_loss / static_cast<double>(test_inputs.size()) << std::endl;
    }
};

template <int num_of_datapoints = 100, double upper_lim = 6.0, double lower_lim = -6.0>
class DataGenerator
{
public:
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> operator()()
    {
        std::vector<std::vector<double>> inputs;
        std::vector<std::vector<double>> outputs;

        // Compute step size based on number of data points
        double step_size = (upper_lim - lower_lim) / (num_of_datapoints - 1);

        for (int i = 0; i < num_of_datapoints; ++i)
        {
            double x = lower_lim + i * step_size;
            inputs.push_back({x});
            outputs.push_back({std::sin(x)});
        }

        return std::make_pair(inputs, outputs);
    }
};

main()
{
    Model nn;
    constexpr int num_of_training_points = 1000;
    constexpr int num_of_testing_points = 50;

    // Generate training data
    auto training_data = DataGenerator<num_of_training_points>()();
    // Generate testing data
    auto testing_data = DataGenerator<num_of_testing_points>()();

    // Open file to save predictions after each epoch
    std::ofstream plot_file("predictions_epoch.csv");
    plot_file << "Epoch,Input,Prediction,Actual\n";

    // Train the neural network using a custom training loop to save data for each epoch
    int epochs = 1000;
    double learning_rate = 0.01;
    for (int e = 0; e < epochs; ++e)
    {
        double total_loss = 0.0;
        for (int i = 0; i < num_of_training_points; ++i)
        {
            nn.forward(training_data.first[i]);
            total_loss += nn.compute_loss(training_data.second[i]);
            nn.backward(training_data.second[i], learning_rate);
        }

        // Save predictions after each epoch
        std::cout << "Epoch " << e + 1 << ", Loss: " << total_loss / static_cast<double>(num_of_training_points) << std::endl;
        for (int i = 0; i < num_of_testing_points; ++i)
        {
            std::vector<double> input = testing_data.first[i];
            std::vector<double> output = nn.predict(input);
            double actual_value = testing_data.second[i][0];
            plot_file << (e + 1) << "," << input[0] << "," << output[0] << "," << actual_value << "\n";
        }
    }
    plot_file.close();

    nn.evaluate(testing_data.first, testing_data.second);

    return 0;
}
