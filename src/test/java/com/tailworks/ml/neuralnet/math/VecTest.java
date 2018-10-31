package com.tailworks.ml.neuralnet.math;

import org.junit.Test;

import static java.lang.Math.max;
import static org.junit.Assert.assertEquals;

public class VecTest {

    @Test
    public void dot() {
        Vec v = new Vec(1, 2, 3);
        Vec u = new Vec(3, -4, 5);
        assertEquals(v.dot(u), 10, 0.01);
        assertEquals(u.dot(v), 10, 0.01);
    }


    @Test
    public void sanityTestEvalutionOrder() {
        Vec a = new Vec(1, 2);
        Vec b = new Vec(2, 3);
        Vec c = new Vec(-1, 4);

        Vec result = a.subtract(b).elementProduct(c.map(x -> max(0, x)));
        assertEquals(result.getData()[0], 0, 0.001);  // (1-2) * 0
        assertEquals(result.getData()[1], -4, 0.001);  // (2-3) * 4
    }

    @Test
    public void test_cMultiply() {
        Vec a = new Vec(1, 2, 3);
        Vec b = new Vec(1, 2, 3, 4, 5, 6);

        Matrix result = a.outerProduct(b);
        assertEquals(result.numberOfRows(), 6);
        assertEquals(result.numberOfCols(), 3);

        assertEquals(result.getData()[0][0], 1, 0.001);
        assertEquals(result.getData()[0][2], 3, 0.001);
        assertEquals(result.getData()[2][0], 3, 0.001);
        assertEquals(result.getData()[5][0], 6, 0.001);
        assertEquals(result.getData()[5][2], 18, 0.001);
    }

    @Test
    public void test_crossMultiply1() {
        Vec a = new Vec(1, 2);

        Matrix m = new Matrix(new double[][]{{-1, 4}, {3, -7}});
        Matrix result = a.columnMultiply(m);

        assertEquals(result.numberOfCols(), 2);
        assertEquals(result.numberOfRows(), 2);

        double[][] data = result.getData();
        assertEquals(data[0][0], -1, 0.01);
        assertEquals(data[0][1], 4, 0.01);
        assertEquals(data[1][0], 6, 0.01);
        assertEquals(data[1][1], -14, 0.01);
    }

    @Test
    public void test_crossMultiply2() {
        Vec a = new Vec(.13849856162855698, -0.03809823651655623);

        Matrix m = new Matrix(new double[][]{{0.40, 0.45}, {0.50, 0.55}});
        Matrix result = a.columnMultiply(m);

        double[][] data = result.getData();
        assertEquals(data[0][0], 0.05539942465142279, 0.01);
        assertEquals(data[0][1], 0.06232435273285064, 0.01);
        assertEquals(data[1][0], -0.019049118258278114, 0.01);
        assertEquals(data[1][1], -0.02095403008410593, 0.01);
    }

    @Test
    public void sub() {
        assertEquals(new Vec(-1, -4, 1), new Vec(1, -2, 3).subtract(new Vec(2, 2, 2)));
    }

    @Test
    public void add() {
        assertEquals(new Vec(3, -4, 1), new Vec(1, -2, 3).add(new Vec(2, -2, -2)));
    }

    @Test
    public void index() {
        assertEquals(3, new Vec(1, -2, 3, 5, -25).indexOfLargestElement());
    }
}
