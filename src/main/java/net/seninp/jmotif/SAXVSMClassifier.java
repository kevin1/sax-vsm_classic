package net.seninp.jmotif;

import java.io.IOException;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;
import net.seninp.jmotif.sax.SAXException;
import net.seninp.jmotif.text.Params;
import net.seninp.jmotif.text.TextProcessor;
import net.seninp.jmotif.text.WordBag;
import net.seninp.util.StackTrace;
import net.seninp.util.UCRUtils;
import org.slf4j.LoggerFactory;
import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import com.beust.jcommander.JCommander;

/**
 * This implements a classifier.
 * 
 * @author psenin
 * 
 */
public class SAXVSMClassifier {

  private static final DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.US);
  private static DecimalFormat fmt = new DecimalFormat("0.00###", otherSymbols);
  private static final Object CR = "\n";
  private static final String COMMA = ", ";

  private static TextProcessor tp = new TextProcessor();

  // double[]: One dimension of a time series
  // List<double[]>: One dimension of all time series of a class
  // Map<String, List<double[]>>: Map of class name to 1D all TS of that class
  // List<Map<...>>: A map for each dimension
  private static List<Map<String, List<double[]>>> trainData = new ArrayList<>();
  private static List<Map<String, List<double[]>>> testData = new ArrayList<>();

  // static block - we instantiate the logger
  //
  private static final Logger consoleLogger;
  private static final Level LOGGING_LEVEL = Level.INFO;

  static {
    consoleLogger = (Logger) LoggerFactory.getLogger(SAXVSMClassifier.class);
    consoleLogger.setLevel(LOGGING_LEVEL);
  }

  private static void printDataStatistics(List<Map<String, List<double[]>>> data, String description) {
    consoleLogger.info(description + " classes count: " + data.get(0).size());
    for (Entry<String, List<double[]>> e : data.get(0).entrySet()) {
      consoleLogger.info("  class " + e.getKey() + ", samples " + e.getValue().size());
      for (int sampleIdx = 0; sampleIdx < e.getValue().size(); sampleIdx++) {
        StringBuilder lengths = new StringBuilder();
        for (int dimIdx = 0; dimIdx < data.size(); dimIdx++) {
          lengths.append(data.get(dimIdx).get(e.getKey()).get(sampleIdx).length).append(" ");
        }
        consoleLogger.info("    sample dim lengths " + lengths);
      }
    }
  }

  public static void main(String[] args) throws SAXException{

    try {
      
      SAXVSMClassifierParams params = new SAXVSMClassifierParams();
      JCommander jct = new JCommander(params, args);

      if (0 == args.length) {
        jct.usage();
        System.exit(-10);
      }

      StringBuffer sb = new StringBuffer(1024);
      sb.append("SAX-VSM Classifier").append(CR);
      sb.append("parameters:").append(CR);

      sb.append("  train data:                  ").append(SAXVSMClassifierParams.TRAIN_FILE).append(CR);
      sb.append("  test data:                   ").append(SAXVSMClassifierParams.TEST_FILE).append(CR);
      sb.append("  num dimensions:              ").append(SAXVSMClassifierParams.NUM_DIMENSIONS).append(CR);
      sb.append("  SAX sliding window size:     ").append(SAXVSMClassifierParams.SAX_WINDOW_SIZE).append(CR);
      sb.append("  SAX PAA size:                ").append(SAXVSMClassifierParams.SAX_PAA_SIZE).append(CR);
      sb.append("  SAX alphabet size:           ").append(SAXVSMClassifierParams.SAX_ALPHABET_SIZE).append(CR);
      sb.append("  SAX numerosity reduction:    ").append(SAXVSMClassifierParams.SAX_NR_STRATEGY).append(CR);
      sb.append("  SAX normalization threshold: ").append(SAXVSMClassifierParams.SAX_NORM_THRESHOLD).append(CR);

      for (int i = 0; i < SAXVSMClassifierParams.NUM_DIMENSIONS; i++) {
        trainData.add(UCRUtils.readUCRData(SAXVSMClassifierParams.TRAIN_FILE + i + ".txt"));
        testData.add(UCRUtils.readUCRData(SAXVSMClassifierParams.TEST_FILE + i + ".txt"));
      }

      printDataStatistics(trainData, "train");
      printDataStatistics(testData, "test");
    }
    catch (Exception e) {
      System.err.println("There was an error...." + StackTrace.toString(e));
      System.exit(-10);
    }
    Params params = new Params(SAXVSMClassifierParams.SAX_WINDOW_SIZE,
        SAXVSMClassifierParams.SAX_PAA_SIZE, SAXVSMClassifierParams.SAX_ALPHABET_SIZE,
        SAXVSMClassifierParams.SAX_NORM_THRESHOLD, SAXVSMClassifierParams.SAX_NR_STRATEGY);
    classify(params);
  }

  private static void classify(Params params) throws SAXException {
    // making training bags collection
    List<List<WordBag>> bags = new ArrayList<>();
    for (Map<String, List<double[]>> dim : trainData) {
      bags.add(tp.labeledSeries2WordBags(dim, params));
    }

    // getting TFIDF done
    HashMap<String, List<HashMap<String, Double>>> tfidfs = new HashMap<>();
    // Iterate through all dimensions
    for (int dimIdx = 0; dimIdx < bags.size(); dimIdx++) {
      List<WordBag> dimBags = bags.get(dimIdx);
      // Iterate through tfidf for each class of this dimension
      for (Entry<String, HashMap<String, Double>> tfidf : tp.computeTFIDF(dimBags).entrySet()) {
        if (!tfidfs.containsKey(tfidf.getKey())) {
          List<HashMap<String, Double>> l = new ArrayList<>();
          for (int i = 0; i < bags.size(); i++) l.add(null);
          tfidfs.put(tfidf.getKey(), l);
        }
        tfidfs.get(tfidf.getKey()).set(dimIdx, tfidf.getValue());
      }
    }

    // classifying
    int testSampleSize = 0;
    int positiveTestCounter = 0;
    // For each class
    for (String label : tfidfs.keySet()) {
      List<double[]> testD = testData.get(0).get(label);
      // For each trace in this class
      for (int i = 0; i < testD.size(); i++) {
        List<double[]> allDims = new ArrayList<>();
        for (int dimIdx = 0; dimIdx < testData.size(); dimIdx++) {
          allDims.add(testData.get(dimIdx).get(label).get(i));
        }
        positiveTestCounter = positiveTestCounter
            + tp.classify(label, allDims, tfidfs, params);
        testSampleSize++;
      }
    }

    // accuracy and error
    double accuracy = (double) positiveTestCounter / (double) testSampleSize;
    double error = 1.0d - accuracy;

    // report results
    System.out.println("classification results: " + toLogStr(params, accuracy, error));

  }

  protected static String toLogStr(Params params, double accuracy, double error) {
    StringBuffer sb = new StringBuffer();
    sb.append("strategy ").append(params.getNrStartegy().toString()).append(COMMA);
    sb.append("window ").append(params.getWindowSize()).append(COMMA);
    sb.append("PAA ").append(params.getPaaSize()).append(COMMA);
    sb.append("alphabet ").append(params.getAlphabetSize()).append(COMMA);
    sb.append(" accuracy ").append(fmt.format(accuracy)).append(COMMA);
    sb.append(" error ").append(fmt.format(error));
    return sb.toString();
  }

}
