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
        if (u.dimension() != dimension())
            throw new IllegalArgumentException(format("Vectors of different dim: Input is %d, Vec is %d", u.dimension(), dimension()));

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
        if (u.dimension() != dimension())
            throw new IllegalArgumentException(format("Vectors of different dim: Input is %d, Vec is %d", u.dimension(), dimension()));

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

    public Vec negate() {
        return scale(-1);
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
        if (u.dimension() != dimension())
            throw new IllegalArgumentException(format("Vectors of different dim: Input is %d, Vec is %d", u.dimension(), dimension()));

        double[] result = new double[u.dimension()];

        for (int i = 0; i < data.length; i++)
            result[i] = data[i] * u.data[i];

        return new Vec(result);
    }

    public Vec copy() {
        return new Vec(data);
    }

    public Vec add(Vec u) {
        if (u.dimension() != dimension())
            throw new IllegalArgumentException(format("Vectors of different dim: Input is %d, Vec is %d", u.dimension(), dimension()));

        double[] result = new double[u.dimension()];

        for (int i = 0; i < data.length; i++)
            result[i] = data[i] + u.data[i];

        return new Vec(result);
    }


    public Matrix columnMultiply(Matrix m) {
        if (dimension() != m.numberOfRows())
            throw new IllegalArgumentException(format("Vector of different dimension than number of rows in matrix: Input is %d, Vec is %d", m.numberOfRows(), dimension()));

        double[][] mData = m.getData();
        double[][] result = new double[dimension()][m.numberOfCols()];

        for (int col = 0; col < m.numberOfCols(); col++)
            for (int row = 0; row < m.numberOfRows(); row++)
                result[row][col] = mData[row][col] * data[row];

        return new Matrix(result);
    }
}
