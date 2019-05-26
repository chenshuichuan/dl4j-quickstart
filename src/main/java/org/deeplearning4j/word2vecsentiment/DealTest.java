package org.deeplearning4j.word2vecsentiment;/**
 * Created by:Ricardo
 * Description:
 * Date: 2019/5/25
 * Time: 11:12
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import org.apache.commons.lang.StringUtils;
/**
 *@ClassName: DealTest
 *@Description: TODO
 *@Author: Ricardo
 *@Date: 2019/5/25 11:12
 **/
public class DealTest {

  private List<TestData> testDataList ;

  public DealTest(String testDataPath) {
    testDataList = new ArrayList<>();
    File file = new File(testDataPath);
    Scanner sc = null;
    try {
      sc = new Scanner(file);
      int lineNum=0;
      while (sc.hasNextLine()) {
        lineNum++;
        String dataLine = sc.nextLine();
        if (dataLine != null) {
          int i = dataLine.indexOf(",");
          if(i>0&&i<dataLine.length()-1){
            String id = dataLine.substring(0,i);
            String review = dataLine.substring(i+1,dataLine.length());
            if(StringUtils.isNumeric(id)){
              testDataList.add(new TestData(Integer.parseInt(id),review));
            }else {
              System.out.println(lineNum+"error:"+dataLine);
            }
          }
          else {
            System.out.println(lineNum+"error:"+dataLine);
          }
        }
      }
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }

  public List<TestData> getTestDataList() {
    return testDataList;
  }

  public void setTestDataList(List<TestData> testDataList) {
    this.testDataList = testDataList;
  }
  public void setTestDataPositiveRate(int index,double positiveRate){
    testDataList.get(index).setPositiveRate(positiveRate);
  }
  public String getTestDataReview(int index){
    return testDataList.get(index).getReview();
  }
  public String[] toStrings(int index){
    if(index>=testDataList.size()){
      return null;
    }
    String[] strings = new String[2];
    strings[0]=testDataList.get(index).getId().toString();
    strings[1]=String.valueOf(testDataList.get(index).getPositiveRate());
    return strings;
  }
}
