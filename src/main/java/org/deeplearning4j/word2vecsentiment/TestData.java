package org.deeplearning4j.word2vecsentiment;/**
 * Created by:Ricardo
 * Description:
 * Date: 2019/5/25
 * Time: 11:14
 */

/**
 *@ClassName: TestData
 *@Description: TODO
 *@Author: Ricardo
 *@Date: 2019/5/25 11:14
 **/
public class TestData {
  private Integer id;
  private String review;
  private double positiveRate;
  public TestData() {

  }
  public TestData(Integer id, String review) {
    this.id = id;
    this.review = review;
    this.positiveRate = 0;
  }
  public TestData(Integer id, String review, double positiveRate) {
    this.id = id;
    this.review = review;
    this.positiveRate = positiveRate;
  }
  public Integer getId() {
    return id;
  }

  public void setId(Integer id) {
    this.id = id;
  }

  public String getReview() {
    return review;
  }

  public void setReview(String review) {
    this.review = review;
  }

  public double getPositiveRate() {
    return positiveRate;
  }

  public void setPositiveRate(double positiveRate) {
    this.positiveRate = positiveRate;
  }
  public String[] toStrings(){

    String[] strings = new String[2];
    strings[0]=getId().toString();
    strings[1]=String.valueOf(getPositiveRate());
    return strings;
  }
}
