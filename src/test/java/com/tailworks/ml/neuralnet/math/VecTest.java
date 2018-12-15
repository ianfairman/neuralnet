package com.tailworks.ml.neuralnet.math;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class VecTest {

    public static final double EPS = 0.000001;

    @Test
    public void dot() {
        Vec v = new Vec(1, 2, 3);
        Vec u = new Vec(3, -4, 5);
        assertEquals(v.dot(u), 10, EPS);
        assertEquals(u.dot(v), 10, EPS);
    }


    @Test
    public void test_cMultiply() {
        Vec a = new Vec(1, 2, 3);
        Vec b = new Vec(1, 2, 3, 4, 5, 6);

        Matrix result = a.outerProduct(b);
        assertEquals(result.rows(), 6);
        assertEquals(result.cols(), 3);

        assertEquals(result.getData()[0][0], 1, EPS);
        assertEquals(result.getData()[0][2], 3, EPS);
        assertEquals(result.getData()[2][0], 3, EPS);
        assertEquals(result.getData()[5][0], 6, EPS);
        assertEquals(result.getData()[5][2], 18, EPS);
    }

    @Test
    public void test_multiply() {
        Vec v = new Vec(1, 2);  // 1x2
        Matrix m = new Matrix(new double[][]{{2, 1, 3}, {3, 4, -1}});  // 2x3
        Vec res = v.mul(m);

        assertEquals(res.dimension(), 3);
        assertEquals(res.getData()[0], 8, EPS);
        assertEquals(res.getData()[1], 9, EPS);
        assertEquals(res.getData()[2], 1, EPS);
    }


    @Test
    public void sub() {
        assertEquals(new Vec(-1, -4, 1), new Vec(1, -2, 3).sub(new Vec(2, 2, 2)));
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
