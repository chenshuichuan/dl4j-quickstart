package org.deeplearning4j.utilities;

/**
 *@ClassName: Params
 *@Description: TODO
 *@Author: Ricardo
 *@Date: 2019/5/26 14:41
 **/
public class Params {

  public static final int minWordFrequency =1;
  public static final int iterations = 5;
  public static final int windowSize = 5;


  /**
   * Location to save and extract the training/testing data
   */
  public static final String DATA_PATH = "E:\\chenyuan\\dataSets\\20190520_train.csv";
  public static final String Test_DATA_PATH = "E:\\chenyuan\\dataSets\\20190520_test.csv";
  public static final String MODAL_PATH = "E:\\chenyuan\\dataSets\\trained_model.zip";
  /** Location (local file system) for the Google News vectors. Set this manually. */
  public static final String WORD_VECTORS_PATH = "E:\\chenyuan\\dataSets\\VectorModal50-5.bin";

  public static final int batchSize = 64;     //Number of examples in each minibatch
  public static final int vectorSize = 50;   //Size of the word vectors. 300 in the Google News model
  public static final int nEpochs = 10;        //Number of epochs (full passes of training data) to train on
  public static final int truncateReviewsToLength = 50;  //Truncate reviews with length (# words) greater than this
  public static final int seed = 0;     //Seed for reproducibility


  public static final String resultCsvPath = "E:\\chenyuan\\dataSets\\result.csv";

}
