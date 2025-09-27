package uo.ml.neural.tracks.train.model;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Factory class for creating neural network models for GPS track analysis.
 * Builds LSTM-based models for predicting track characteristics.
 */
public class ModelFactory {
    
    /**
     * Creates a standard LSTM model for GPS track sequence prediction.
     * 
     * Architecture:
     * - LSTM layer (128 units, tanh activation)
     * - Global pooling (LAST - uses mask to get last real timestep)
     * - Dense layer (64 units, ReLU activation)  
     * - Output layer (3 units, linear activation for regression)
     * 
     * @param nFeatures Number of input features per timestep (typically 3: dh, dz, slope)
     * @param learningRate Learning rate for Adam optimizer (typically 1e-3)
     * @return Configured MultiLayerNetwork ready for training
     */
    public static MultiLayerNetwork createLSTMModel(int nFeatures, double learningRate) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(learningRate))
            .list()
            .layer(0, new LSTM.Builder()
                .nIn(nFeatures)
                .nOut(128)
                .activation(Activation.TANH)
                .build())
            .layer(1, new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .build())
            .layer(2, new DenseLayer.Builder()
                .nOut(64)
                .activation(Activation.RELU)
                .build())
            .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
                .nOut(3)
                .activation(Activation.IDENTITY)
                .build())
            .setInputType(InputType.recurrent(nFeatures))
            .build();
        
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        return model;
    }
    
    /**
     * Creates a standard LSTM model with default learning rate.
     * 
     * @param nFeatures Number of input features per timestep
     * @return Configured MultiLayerNetwork with learning rate 1e-3
     */
    public static MultiLayerNetwork createLSTMModel(int nFeatures) {
        return createLSTMModel(nFeatures, 1e-3);
    }
    
    /**
     * Creates a deeper LSTM model for more complex patterns.
     * 
     * @param nFeatures Number of input features per timestep
     * @param learningRate Learning rate for Adam optimizer
     * @return Deeper MultiLayerNetwork with 2 LSTM layers
     */
    public static MultiLayerNetwork createDeepLSTMModel(int nFeatures, double learningRate) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(learningRate))
            .list()
            .layer(0, new LSTM.Builder()
                .nIn(nFeatures)
                .nOut(128)
                .activation(Activation.TANH)
                .build())
            .layer(1, new LSTM.Builder()
                .nIn(128)
                .nOut(64)
                .activation(Activation.TANH)
                .build())
            .layer(2, new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .build())
            .layer(3, new DenseLayer.Builder()
                .nOut(32)
                .activation(Activation.RELU)
                .build())
            .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
                .nOut(3)
                .activation(Activation.IDENTITY)
                .build())
            .setInputType(InputType.recurrent(nFeatures))
            .build();
        
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        return model;
    }
}