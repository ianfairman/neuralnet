package com.tailworks.ml.neuralnet.math;

import java.util.Arrays;

import static java.lang.String.format;
import static java.util.Arrays.stream;

public class Vec {

    private double[] data;

    public Vec(double... data) {
        this.data = data;
    }

    public Vec(int... data) {
        this(stream(data).asDoubleStream().toArray());
    }

    public Vec(int size) {
        data = new double[size];
    }

    public int dimension() {
        return data.length;
    }

    public double dot(Vec u) {
        assertCorrectDimension(u.dimension());

        double sum = 0;
        for (int i = 0; i < data.length; i++)
            sum += data[i] * u.data[i];

        return sum;
    }

    public Vec map(Function fn) {
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++)
            result[i] = fn.apply(data[i]);
        return new Vec(result);
    }

    public double[] getData() {
        return data;
    }

    @Override
    public String toString() {
        return "Vec{" + "data=" + Arrays.toString(data) + '}';
    }

    public int indexOfLargestElement() {
        int ixOfLargest = 0;
        for (int i = 0; i < data.length; i++)
            if (data[i] > data[ixOfLargest]) ixOfLargest = i;
        return ixOfLargest;
    }

    public Vec subtract(Vec u) {
        assertCorrectDimension(u.dimension());

        double[] result = new double[u.dimension()];

        for (int i = 0; i < data.length; i++)
            result[i] = data[i] - u.data[i];

        return new Vec(result);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Vec vec = (Vec) o;

        return Arrays.equals(data, vec.data);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(data);
    }


    public Vec scale(double s) {
        return map(value -> s * value);
    }

    public Matrix outerProduct(Vec u) {
        double[][] result = new double[u.dimension()][dimension()];

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < u.data.length; j++) {
                result[j][i] = data[i] * u.data[j];
            }
        }
        return new Matrix(result);
    }

    public Vec elementProduct(Vec u) {
        assertCorrectDimension(u.dimension());

        double[] result = new double[u.dimension()];

        for (int i = 0; i < data.length; i++)
            result[i] = data[i] * u.data[i];

        return new Vec(result);
    }

    public Vec copy() {
        return new Vec(data);
    }

    public Vec add(Vec u) {
        assertCorrectDimension(u.dimension());

        double[] result = new double[u.dimension()];

        for (int i = 0; i < data.length; i++)
            result[i] = data[i] + u.data[i];

        return new Vec(result);
    }

    public Vec multiply(Matrix m) {
        assertCorrectDimension(m.numberOfRows());

        double[][] mData = m.getData();
        double[] result = new double[m.numberOfCols()];

        for (int col = 0; col < m.numberOfCols(); col++)
            for (int row = 0; row < m.numberOfRows(); row++)
                result[col] += mData[row][col] * data[row];

        return new Vec(result);
    }


    private void assertCorrectDimension(int inpDim) {
        if (dimension() != inpDim)
            throw new IllegalArgumentException(format("Different dimensions: Input is %d, Vec is %d", inpDim, dimension()));
    }
}
