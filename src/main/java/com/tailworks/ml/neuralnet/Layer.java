package com.tailworks.ml.neuralnet;

import com.tailworks.ml.neuralnet.math.Matrix;
import com.tailworks.ml.neuralnet.math.Vec;


public class Layer {

    private final int size;
    private Vec inData;
    private Vec outData;
    private Activation activation;
    private Matrix weights;
    private Vec bias;

    private transient Matrix deltaWeights;
    private transient Vec deltaBias;
    private transient int weightsAdded = 0;
    private transient int biasesAdded = 0;

    private Layer precedingLayer;


    public Layer(int size, Activation activation) {
        this(size, activation, 0);
    }

    public Layer(int size, Activation activation, double initialBias) {
        this.size = size;
        inData = new Vec(size);
        outData = new Vec(size);
        bias = new Vec(size).map(x -> initialBias);
        deltaBias = new Vec(size);
        this.activation = activation;
    }

    public Layer(int size, Activation activation, Vec bias) {
        this.size = size;
        inData = new Vec(size);
        this.bias = bias.copy();
        deltaBias = new Vec(size);
        this.activation = activation;
    }

    public int size() {
        return size;
    }

    public Layer feedWith(Vec in) {
        if (weights != null) {
            in = in.mul(weights);
        }
        if (in.dimension() != inData.dimension()) throw new IllegalArgumentException();
        System.arraycopy(in.getData(), 0, inData.getData(), 0, inData.getData().length);

        outData = activation.fn(inData.add(bias));
        return this;
    }

    public Vec getOut() {
        return outData;
    }

    public Activation getActivation() {
        return activation;
    }

    public void addWeights(Matrix weights) {
        this.weights = weights;
        deltaWeights = new Matrix(weights.rows(), weights.cols());
    }

    public Matrix getWeights() {
        return weights;
    }

    public Layer getPrecedingLayer() {
        return precedingLayer;
    }

    public void setPrecedingLayer(Layer precedingLayer) {
        this.precedingLayer = precedingLayer;
    }

    public boolean hasPrecedingLayer() {
        return precedingLayer != null;
    }

    public Vec getBias() {
        return bias;
    }


    public synchronized void addDeltaWeights(Matrix dW) {
        deltaWeights.add(dW);
        weightsAdded++;
    }

    public synchronized void addDeltaBias(Vec dB) {
        deltaBias = deltaBias.add(dB);
        biasesAdded++;
    }

    public synchronized void updateWeights(double learningRate) {
        Matrix average_dW = deltaWeights.mul(1.0 / weightsAdded);
        weights.sub(average_dW.mul(learningRate));

        deltaWeights.map(a -> 0);
        weightsAdded = 0;
    }

    public synchronized void updateBias(double learningRate) {
        Vec average_bias = deltaBias.mul(1.0 / biasesAdded);
        bias = bias.sub(average_bias.mul(learningRate));

        deltaBias = deltaBias.map(a -> 0);
        biasesAdded = 0;
    }


    // ------------------------------------------------------------------


    public LayerState getState() {
        return new LayerState(this);
    }

    public static class LayerState {

        double[][] weights;
        double[] bias;
        String activation;

        public LayerState(Layer layer) {
            weights = layer.getWeights() != null ? layer.getWeights().getData() : null;
            bias = layer.getBias().getData();
            activation = layer.activation.getName();
        }
    }
}
