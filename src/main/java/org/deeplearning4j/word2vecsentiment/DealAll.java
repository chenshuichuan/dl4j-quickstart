package org.deeplearning4j.word2vecsentiment;/**
 * Created by:Ricardo
 * Description:
 * Date: 2019/5/26
 * Time: 15:01
 */

/**
 *@ClassName: DealAll
 *@Description: TODO
 *@Author: Ricardo
 *@Date: 2019/5/26 15:01
 **/
public class DealAll {
  public static void main(String[] args) throws Exception {

    Word2VecRawTextExample.run();

    MyWord2VecSentimentRNN.runModal();

    LoadWord2VecSentimentRNN.LoadTestData();
  }
}
