package org.deeplearning4j.word2vecsentiment;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
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
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.utilities.DataUtilities;
import org.deeplearning4j.utilities.Params;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URL;

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
public class MyWord2VecSentimentRNN {
  private static Logger log = LoggerFactory.getLogger(MyWord2VecSentimentRNN.class);
  /**
   * Location to save and extract the training/testing data
   */
  public static final String DATA_PATH = "E:\\chenyuan\\dataSets\\20190520_train.csv";
  public static final String Test_DATA_PATH = "E:\\chenyuan\\dataSets\\20190520_test.csv";

  /** Location (local file system) for the Google News vectors. Set this manually. */
  public static final String WORD_VECTORS_PATH = "E:\\chenyuan\\dataSets\\VectorModal50.bin";


  public static void main(String[] args) throws Exception {
    runModal();
  }
  public static void runModal()throws Exception{
    if (WORD_VECTORS_PATH.startsWith("/PATH/TO/YOUR/VECTORS/")) {
      throw new RuntimeException("Please set the WORD_VECTORS_PATH before running this example");
    }

    int batchSize = Params.batchSize;     //Number of examples in each minibatch
    int vectorSize = Params.vectorSize;   //Size of the word vectors. 300 in the Google News model
    int nEpochs = Params.nEpochs;        //Number of epochs (full passes of training data) to train on
    int truncateReviewsToLength = Params.truncateReviewsToLength;  //Truncate reviews with length (# words) greater than this
    final int seed = Params.seed;     //Seed for reproducibility

    Nd4j.getMemoryManager().setAutoGcWindow(10000);  //https://deeplearning4j.org/workspaces

    //Set up network configuration
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .updater(new Adam(2e-2))
            .l2(1e-5)
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
            .trainingWorkspaceMode(WorkspaceMode.SEPARATE).inferenceWorkspaceMode(WorkspaceMode.SEPARATE)   //https://deeplearning4j.org/workspaces
            .list()
            .layer(0, new LSTM.Builder().nIn(vectorSize).nOut(256)
                    .activation(Activation.TANH).build())
            .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                    .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
            .pretrain(false).backprop(true).build();

    MultiLayerNetwork net = new MultiLayerNetwork(conf);
    net.init();
    net.setListeners(new ScoreIterationListener(1));

    //DataSetIterators for training and testing respectively
    WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));

    SentimentIterator train = new SentimentIterator(Params.DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, true);
    SentimentIterator test = new SentimentIterator(Params.Test_DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false);

    //train.setCursor(4736);
    System.out.println("Starting training");
    for (int i = 0; i < nEpochs; i++) {
      net.fit(train);
      train.reset();
      System.out.println("Epoch " + i + " complete. Starting evaluation:");
    }
    //Run evaluation. This is on 25k reviews, so can take some time
    Evaluation evaluation = net.evaluate(test);
    System.out.println(evaluation.stats());
    log.info("SAVE TRAINED MODEL");
    // Where to save model
    File locationToSave = new File(Params.MODAL_PATH);
    // boolean save Updater
    boolean saveUpdater = false;
    // ModelSerializer needs modelname, saveUpdater, Location
    ModelSerializer.writeModel(net, locationToSave, saveUpdater);

    System.out.println("----- Example complete -----");
  }

}
