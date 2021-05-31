package adv_algorithms;

import Codeforces_654.C;
import edu.princeton.cs.algs4.Complex;
import edu.princeton.cs.algs4.FFT;
import edu.princeton.cs.algs4.Stopwatch;

import java.util.*;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.stream.Collectors;

public class FasterSubsetSum {

    public static void main(String[] args) {
        //setup test thing
        FasterSubsetSum runner = new FasterSubsetSum();
        var a = new HashSet<Integer>();
        int n = 1000;
        var t = 10000;
        Random r = new Random(129L);
        for (int i = 0; i < n; i++) {
            a.add(r.nextInt(t));
        }
        var values = new ArrayList<>(a);
        var delta = 0.00001;
        //var solution = runner.run(a, t, delta);
        Stopwatch timer = new Stopwatch();
        var test = runner.run(values, t, delta);
        long endTime = System.nanoTime();
        System.out.println(test);
        System.out.printf("Solved in %f seconds", timer.elapsedTime());
    }

    List<Integer> run(ArrayList<Integer> Z, int t, double delta) {
        var n = Z.size();
        Z.sort(Collections.reverseOrder());
        var layers = createLayers(Z, t, n);
        final List<Integer>[] S = new List[]{List.of(0)};
        ArrayList<Thread> threads = new ArrayList<>();
        for (int i = 0; i < layers.size(); i++) {
            var layer = layers.get(i);
            if (layer.size() > 0) {
                var Si = ColorCodingLayer(layer, t, Math.pow(2, i + 1)-1, delta / layers.size());
                System.out.println("Thread run done");
                S[0] = SumSet(S[0], Si, t);
                /**
                int finalI = i;
                Thread r = new Thread(() -> {
                    System.out.println("Starting thread");
                    var Si = ColorCodingLayer(layer, t, Math.pow(2, finalI + 1)-1, delta / layers.size());
                    System.out.println("Thread run done");
                    synchronized (this) {
                        S[0] = SumSet(S[0], Si, t);
                    }
                });
                threads.add(r);
                 **/
            }
        }
        /**
        for (Thread r : threads) {
            r.start();
        }
        for (Thread r : threads) {
            try {
                r.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
         **/
        return S[0];
    }

    private int roundToPowerOf2(double s) {
        return (int) Math.pow(2, Math.ceil(log2(s)));
    }

    private ArrayList<Integer>  SumSet(List<Integer> A, List<Integer> B, int t) {
        int length = A.stream().max(Integer::compareTo).get() + B.stream().max(Integer::compareTo).get() + 1;
        length = roundToPowerOf2(length);
        ArrayList<Complex> AComplex = new ArrayList<>();
        ArrayList<Complex> BComplex = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            AComplex.add(new Complex(0,0));
            BComplex.add(new Complex(0,0));
        }
        AComplex.set(0, new Complex(1, 0));
        BComplex.set(0, new Complex(1, 0));
        for (Integer a : A) {
            AComplex.set(a, new Complex(1, 0));
        }
        for (Integer b : B) {
            BComplex.set(b, new Complex(1, 0));
        }
        var convolution = FFT.convolve(AComplex.toArray(Complex[]::new), BComplex.toArray(Complex[]::new));
        ArrayList<Integer> solution = new ArrayList<>();
        for (int i = 0; i < Math.min(convolution.length, t); i++) {
            if (convolution[i].re() > 0.01) {
                solution.add(i);
            }
        }
        return solution;
    }

    private List<Integer> ColorCodingLayer(ArrayList<Integer> Z, int t, double l, double delta) {
        var divisor = log2(l / delta);
        if (l < divisor) {
            return ColorCoding(Z, t, l, delta);
        }
        var m = roundToPowerOf2(l / divisor);
        var partitions = partitionSetIntoK(Z, m);
        m = roundToPowerOf2(partitions.size());
        while (partitions.size() < m) {
            partitions.add(List.of(1));
        }
        var gamma = 6* divisor;
        if (gamma > l) {
            gamma = l;
        }
        var S = new ArrayList<List<Integer>>();
        for (int j = 0; j < m; j++) {
            S.add(ColorCoding(partitions.get(j), t, gamma, delta / l));
        }
        for (int h = 1; h <= log2(m); h++) {
            var threshold = (int) Math.ceil(Math.min(Math.pow(2, h) * 2 * gamma * t / l, t));
            for (int j = 1; j <= (int) (m / Math.pow(2, h)) ; j++) {
                S.set(j-1, SumSet(S.get(2*j-1-1), S.get(2*j-1), threshold));
            }
        }
        return S.get(0);
    }

    private ArrayList<Integer> ColorCoding(List<Integer> Z, int t, double k, double delta) {
        int repetitions = 1;//(int) Math.ceil(log(1.0 / delta, 4.0/3.0));
        var S = new ArrayList<List<Integer>>();
        for (int i = 0; i < repetitions; i++) {
            var partition = partitionSetIntoK(Z, (int) (k*k));
            var sumset = partition.parallelStream().reduce(List.of(1), (integers, integers2) -> SumSet(integers, integers2, t));
            S.add(sumset);
        }
        var union = new HashSet<Integer>();
        for (List<Integer> sumset : S) {
            union.addAll(sumset);
        }
        return new ArrayList<>(union);
    }

    private List<List<Integer>> partitionSetIntoK(List<Integer> Z, int k) {
        HashMap<Integer, ArrayList<Integer>> partitions = new HashMap<>();
        Random r = new Random();
        for (Integer z : Z) {
            int goesTo = r.nextInt(k + 1);
            partitions.putIfAbsent(goesTo, new ArrayList<>());
            partitions.get(goesTo).add(z);
        }
        return partitions.values().stream().filter(integers -> integers.size() > 0).collect(Collectors.toCollection(ArrayList::new));
    }

    private ArrayList<ArrayList<Integer>> createLayers(List<Integer> z, int t, int n) {
        ArrayList<ArrayList<Integer>> partitions = new ArrayList<>();
        int nextIndex = 0;
        for (int i = 1; i < Math.ceil(log2(n)); i++) {
            var partition = new ArrayList<Integer>();
            while (nextIndex < n &&
                    z.get(nextIndex) > t /Math.pow(2, i) &&
                    z.get(nextIndex) < t /Math.pow(2, i-1)) {
                partition.add(z.get(nextIndex));
                nextIndex++;
            }
            partitions.add(partition);
        }
        var partition = new ArrayList<Integer>();
        for (int i = nextIndex; nextIndex < n; nextIndex++) {
            partition.add(z.get(nextIndex));
        }
        partitions.add(partition);
        return partitions;
    }

    double log2(double x) {
        return Math.log(x) / Math.log(2);
    }

    double log(double x, double base) {
        return Math.log(x) / Math.log(base);
    }



}
