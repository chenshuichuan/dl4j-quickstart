package org.deeplearning4j.word2vecsentiment;

import com.csvreader.CsvWriter;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.utilities.CSVUtils;
import org.deeplearning4j.utilities.DataUtilities;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

/**
 * Example: Given a movie review (raw text), classify that movie review as either positive or negative based on the words it contains.
 * This is done by combining Word2Vec vectors and a recurrent neural network model. Each word in a review is vectorized
 * (using the Word2Vec model) and fed into a recurrent neural network.
 * Training data is the "Large Movie Review Dataset" from http://ai.stanford.edu/~amaas/data/sentiment/
 * This data set contains 25,000 training reviews + 25,000 testing reviews
 * <p>
 * Process:
 * 1. Automatic on first run of example: Download data (movie reviews) + extract
 * 2. Load existing Word2Vec model (for example: Google News word vectors. You will have to download this MANUALLY)
 * 3. Load each each review. Convert words to vectors + reviews to sequences of vectors
 * 4. Train network
 * <p>
 * With the current configuration, gives approx. 83% accuracy after 1 epoch. Better performance may be possible with
 * additional tuning.
 * <p>
 * NOTE / INSTRUCTIONS:
 * You will have to download the Google News word vector model manually. ~1.5GB
 * The Google News vector model available here: https://code.google.com/p/word2vec/
 * Download the GoogleNews-vectors-negative300.bin.gz file
 * Then: set the WORD_VECTORS_PATH field to point to this location.
 *
 * @author Alex Black
 */
public class LoadWord2VecSentimentRNN {
  private static Logger log = LoggerFactory.getLogger(LoadWord2VecSentimentRNN.class);

  /**
   * Location to save and extract the training/testing data
   */
  public static final String DATA_PATH = "E:\\chenyuan\\dataSets\\20190520_train.csv";
  public static final String resultCsvPath = "E:\\chenyuan\\dataSets\\result.csv";
  public static final String Test_DATA_PATH = "E:\\chenyuan\\dataSets\\20190520_test.csv";

  /**
   * Location (local file system) for the Google News vectors. Set this manually.
   */
  public static final String WORD_VECTORS_PATH = "E:\\chenyuan\\dataSets\\VectorModal50.bin";


  public static void main(String[] args) throws Exception {
    if (WORD_VECTORS_PATH.startsWith("/PATH/TO/YOUR/VECTORS/")) {
      throw new RuntimeException("Please set the WORD_VECTORS_PATH before running this example");
    }

    int batchSize = 64;     //Number of examples in each minibatch
    int vectorSize = 50;   //Size of the word vectors. 300 in the Google News model
    int nEpochs = 1;        //Number of epochs (full passes of training data) to train on
    int truncateReviewsToLength = 50;  //Truncate reviews with length (# words) greater than this
    final int seed = 0;     //Seed for reproducibility

    Nd4j.getMemoryManager().setAutoGcWindow(10000);  //https://deeplearning4j.org/workspaces
    // Where to save model
    File locationToSave = new File(DATA_PATH + "trained_model.zip");
    MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
    net.getLabels();

    //DataSetIterators for training and testing respectively
    WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
    SentimentIterator test = new SentimentIterator(Test_DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false);

    DealTest dealTest = new DealTest(Test_DATA_PATH);
    List<String[]> stringsList = new ArrayList<>();

    for (int i =0;i<dealTest.getTestDataList().size();i++){
      String firstPositiveReview = dealTest.getTestDataReview(i);
      INDArray features = test.loadFeaturesFromString(firstPositiveReview, truncateReviewsToLength);
      INDArray networkOutput = net.output(features);
      int timeSeriesLength = (int) networkOutput.size(2);
      INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

      dealTest.setTestDataPositiveRate(i,probabilitiesAtLastWord.getDouble(0));
      //System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
      //System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));
      String[] strings= dealTest.getTestDataList().get(i).toStrings();
      stringsList.add(strings);
    }

    System.out.println("----- Example complete -----");
    CSVUtils.writeCsvFile(resultCsvPath,dealTest.getTestDataList());
    System.out.println("----- writeCsv complete -----");
  }


}
