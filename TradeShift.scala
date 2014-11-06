
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS,LogisticRegressionWithSGD}
import org.apache.spark.mllib.optimization.{L1Updater,SquaredL2Updater}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vectors,Vector}
import org.apache.spark.mllib.util.MLUtils

import org.apache.spark.mllib.feature.StandardScaler
import org.apache.log4j.Logger
import org.apache.log4j.Level


object TradeShift {
	def main(args: Array[String]) {

Logger.getLogger("org").setLevel(Level.WARN)
Logger.getLogger("akka").setLevel(Level.WARN)
	
	val trPara=args(1)
		println("Start:")
		println(trPara)

		val master = "spark://wh002:6066"
		
		val conf = new SparkConf()
			.setMaster(master)
			.setAppName("kaggle:"+trPara)
			.set("spark.executor.memory", "36g")
			

		val sc = new SparkContext(conf)


val logloss=(p:Double,y:Double)=> - math.log(   math.min(math.max(if (y==1) p else 1.0-p,1e-15) ,1.0-1e-15)  )
val tr=(0 until 10).map{i=>sc.textFile("/wang/datasets/tradeshift/cv/"+i).map(_.split(",",-1))}
val te=sc.textFile("/wang/datasets/tradeshift/test.csv").filter(! _.startsWith("id,") ).map(_.split(",",-1))

val nY=33
val nFeatures=145
val xb=1
val CatNum=List(14, 16, 17, 21, 22, 26, 45, 47, 48, 52, 53, 57, 75, 77, 78, 82, 83, 87, 105, 107, 108, 112, 113, 117, 130, 132, 133, 137, 138, 142).map(_+xb)
val Cat=List(0,1,2,3,9,10,11,12,13,23,24,25,29,30,31,32,33,34,40,41,42,43,44,54,55,56,60,61,62,63,64,70,71,72,73,74,84,85,86,90,91,92,93,94,100,101,102,103,104,114,115,116,125,126,127,128,129,139,140,141).map(_+xb)
val Num=Array(4,5,6,7,8,14,15,16,17,18,19,20,21,22,26,27,28,35,36,37,38,39,45,46,47,48,49,50,51,52,53,57,58,59,65,66,67,68,69,75,76,77,78,79,80,81,82,83,87,88,89,95,96,97,98,99,105,106,107,108,109,110,111,112,113,117,118,119,120,121,122,123,124,130,131,132,133,134,135,136,137,138,142,143,144).map(_+xb)
val allCat=(CatNum:::Cat)
val listY=( (xb+nFeatures) until  (xb+nFeatures+nY))


val ca=sc.textFile("/wang/tmp/cat1").coalesce(1).map(_.split(",")).map(v=>v(0).toInt->v(1)).filter(_._1>1).map(_._2).collect.zipWithIndex
val cm=ca.toMap
val nCat=ca.length+allCat.length
val catList=(v:Array[String])=>allCat.map{c=>cm.get( (c+nY).toString+":"+v(c) )}.zipWithIndex.map{case (v,i)=>if (v==None) ca.length+i.toInt else v.get}


 
 
val t1=tr.map{r=>r.map{v=>
val  y=listY.map { i=>v(i).toDouble}
val num=Vectors.dense(Num.map{i=>v(i).toDouble})
val cval=catList(v)
(y,num,cval)}}

val te1=te.map{v=>
val id=v(0).toInt
val num=Vectors.dense(Num.map{i=>v(i).toDouble})
val cval=catList(v)
(id,num,cval)}



val train=sc.union(t1.slice(0,9))

val scaler = new StandardScaler(withMean = true, withStd = true).fit( train.map{case (_,x,_) => x})
 
val trans=(num:org.apache.spark.mllib.linalg.Vector, cval:List[Int])=>
Vectors.sparse( Num.length+nCat  ,Array.concat(  //+allCat.length*32
scaler.transform(num).toArray.zipWithIndex.map(v=>v._2->v._1) 
, cval.zipWithIndex.toArray.map{case( v,i)=>v+Num.length->1.0} 
//,  cval.zipWithIndex.toArray.flatMap{case( v,i)=>gm(v).zipWithIndex.map{case (g,j)=>i*32+j+Num.length->g}} //+nCat
) ) 


val tr2=train.map{
 case (y,num,cval)=> 
 (y,  trans(num,cval)) 
 }.repartition(360).cache
val test=t1(9).map{
 case (y,num,cval)=> 
 (y,  trans(num,cval)) 
 }.cache
val te2=te1.map{
 case (id,num,cval)=> 
 (id,  trans(num,cval)) 
 }.cache

 
 val buildAlg=(para:String)=>{
 val v=para.split("_",-1)
 val lrAlg = new LogisticRegressionWithLBFGS()
lrAlg.optimizer.
  setNumIterations(v(2).toInt).
  setRegParam(v(1).toDouble).
  setUpdater( if(v(0)=="L1") new L1Updater else new SquaredL2Updater) 
lrAlg  
 }
 

 
 
val Va=(sPara:String,iy:Int)=>{
val training=tr2.map{ case (y,x)=>LabeledPoint( y(iy),x)}
val lrAlg =buildAlg(sPara) 
val model = lrAlg.run(training)
model.clearThreshold()
val score=test.map { case(v,x) =>  (model.predict(x),v(iy) ) }
val loss=score.map{case(p,y)=> logloss(p,y) }
(score,loss)
}


val train_sub=
sc.union(t1).map{
 case (y,num,cval)=> 
 (y,  trans(num,cval)) 
 }.repartition(360).cache
 
val Submit=(sPara:String,iy:Int)=>{

val training=train_sub.map{ case (y,x)=>LabeledPoint( y(iy),x)}
val lrAlg =buildAlg(sPara) 
val model = lrAlg.run(training)
model.clearThreshold()
te2.map { case(idx,x) =>
  val score = model.predict(x)
  idx.toString+"_y"+(iy+1).toString+","+"%.10f".format(score)
}.coalesce(1).saveAsTextFile("/wang/tmp/r/k"+(iy+1).toString ,classOf[org.apache.hadoop.io.compress.GzipCodec])
val loss=tr2
.map { case(v,x) =>  (model.predict(x),v(iy) ) }
.map{case(p,y)=> logloss(p,y)}.mean

loss
}





val t0=System.currentTimeMillis/1000.

if (args.length>2){
for(i<-0 until 33) {
	if (i!=13)
	{
	println( "Time: %.3f :".format(System.currentTimeMillis/1000.-t0))
	println( "Result: %d , %.6f ".format( i,Submit(trPara,i)   ))
	}
}
}
else{
val rr=(0 until 33).filter(_!=13).map{i=>
	
//   val (score,loss)=Va("L1_1e-4_10",32)	
	println( "Time: %.3f :".format(System.currentTimeMillis/1000.-t0))
	val (score,loss)=Va(trPara,i)//"L1_0.001_10"
	val hist=loss.map(math.log _).histogram((-35.0 to 4.0 by 0.1).toArray).toList
	 val sh=score.map(_._1).histogram((0. to 1. by 0.001).toArray).toList
	// score.map(_._1).histogram((0. to 1. by 0.001).toArray)
	 val ml=loss.mean
	println( "Result: %d , %.6f ".format( i,ml   ))
	(i,ml,hist,score.map(_._2).mean,sh)
}


  sc.parallelize(rr,1).map{case(i,ml,hist,sm,sh)=>
  i.toString+","+ml.toString+","+hist.map(_.toString).mkString(":")+","+sm.toString+","+sh.map(_.toString).mkString(":")
  }.saveAsTextFile("/wang/tmp/r/"+trPara)

}

}
}
