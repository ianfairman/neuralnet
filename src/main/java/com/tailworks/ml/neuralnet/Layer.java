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

    private Layer precedingLayer;
    private Vec backpropError;

    public Layer(int size, Activation activation) {
        this.size = size;
        inData = new Vec(size);
        outData = new Vec(size);
        this.activation = activation;
    }

    public Layer(int size, Activation activation, double initialBias) {
        this.size = size;
        inData = new Vec(size);
        outData = new Vec(size);
        bias = new Vec(size).map(x -> initialBias);
        this.activation = activation;
    }

    public Layer(int size, Activation activation, Vec bias) {
        this.size = size;
        inData = new Vec(size);
        this.bias = bias.copy();
        this.activation = activation;
    }

    public int size() {
        return size;
    }

    public Layer feedWith(Vec in) {
        if (weights != null) {
            in = in.multiply(weights);
        }
        if (in.dimension() != inData.dimension()) throw new IllegalArgumentException();
        System.arraycopy(in.getData(), 0, inData.getData(), 0, inData.getData().length);

        outData = activation.fn(bias != null ? inData.add(bias) : inData);
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
    }

    public Matrix getWeights() {
        return weights;
    }

    public Layer getPrecedingLayer() {
        return precedingLayer;
    }

    public void setPrecedingLayer(Layer preceedingLayer) {
        this.precedingLayer = preceedingLayer;
    }

    public boolean hasPrecedingLayer() {
        return precedingLayer != null;
    }

    public void setBackpropError(Vec backpropError) {
        this.backpropError = backpropError;
    }

    public Vec getBackpropError() {
        return backpropError;
    }


    public Vec getBias() {
        return bias;
    }

    public void setBias(Vec bias) {
        this.bias = bias;
    }

    public boolean hasBias() {
        return bias != null;
    }

    public LayerState getState() {
        return new LayerState(this);
    }



    public static class LayerState {

        double[][] weights;
        double[] bias;
        String activation;

        public LayerState(Layer layer) {
            weights = layer.getWeights() != null ? layer.getWeights().getData() : null;
            bias = layer.bias != null ? layer.getBias().getData() : null;
            activation = layer.activation.getName();
        }
    }
}
