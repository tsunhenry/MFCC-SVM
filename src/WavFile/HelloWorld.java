package WavFile;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

import javax.print.DocFlavor.URL;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioFormat.Encoding;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;


import WavFile.WavFile;
import WavFile.WavFileException;

public class HelloWorld {
	
	public static void saveMatrixAsText(String filename, double[][] matrixToSave) throws IOException{
		BufferedWriter out = new BufferedWriter(new FileWriter(filename));
      	for(int i = 0; i < matrixToSave.length; i++) {
      		for(int j = 0; j < matrixToSave[0].length; j++) {
      		        out.write(matrixToSave[i][j] + " ");
      		 }
      		 out.newLine();
      	}    	
        	out.close();
	}
	public static void saveArrayAsText(String filename, double[] arrayToSave) throws IOException{
		BufferedWriter out = new BufferedWriter(new FileWriter(filename));
      	for(int i = 0; i < arrayToSave.length; i++) {
	        out.write(arrayToSave[i] + " ");	 
      	}    	
        	out.close();
	}
	public static double[] readFully(File file)throws UnsupportedAudioFileException, IOException, WavFileException{
	
			WavFile wavFile = WavFile.openWavFile(file);

			int nSamples = (int) wavFile.getNumFrames();	
			System.out.println(nSamples);
			int fs = (int) wavFile.getSampleRate();
			System.out.println(fs);

			int[] signal = new int[nSamples];
			double[] signaldouble = new double[nSamples];

			wavFile.readFrames(signal, nSamples);
			int framesRead;
			
			do {
				// Read frames into buffer
				framesRead = wavFile.readFrames(signal, nSamples);
	
			} while (framesRead != 0);

			// Close the wavFile
			wavFile.close();

			BufferedWriter sigout = new BufferedWriter(new FileWriter("signal.txt"));		
			for (int i = 0; i < nSamples; i++) {
				signaldouble[i] = signal[i];
				//System.out.println(signaldouble[i]);
				sigout.write(signal[i] + " ");
			}
			sigout.close();
			
			/*
			for (int i = 0; i < 30; i++) {
				
				System.out.println(signaldouble[i]);
				
			}
			*/
			return signaldouble;
	}
	public static double[] filterSignal(double[] inputSignal, double filCoeff) {
		 double[] filSig = new double[inputSignal.length];
	        
	        for (int i = 0; i < inputSignal.length; i++) {
	        		if (i == 0)
	        			 filSig[i] = inputSignal[i];
	        		else
	        			filSig[i] = inputSignal[i] - inputSignal[i-1]* filCoeff;
	        }
	        return filSig;
	}
	public static double[] fft(final double[] inputReal, double[] inputImag) {
		
		//https://stackoverflow.com/questions/3287518/reliable-and-fast-fft-in-java
		// - n is the dimension of the problem
		// - nu is its logarithm in base e
		int n = inputReal.length;
		
		// If n is a power of 2, then ld is an integer (_without_ decimals)
		double ld = Math.log(n) / Math.log(2.0);
		
		// Here I check if n is a power of 2. If exist decimals in ld, I quit
		// from the function returning null.
		if (((int) ld) - ld != 0) {
			System.out.println("The number of elements is not a power of 2.");
			return null;
		}
		
		// Declaration and initialization of the variables
		// ld should be an integer, actually, so I don't lose any information in
		// the cast
		int nu = (int) ld;
		int n2 = n / 2;
		int nu1 = nu - 1;
		double[] xReal = new double[n];
		double[] xImag = new double[n];
		double tReal, tImag, p, arg, c, s;
		
		// Here I check if I'm going to do the direct transform or the inverse
		// transform.
		double constant;
		constant = -2 * Math.PI;
		
		// I don't want to overwrite the input arrays, so here I copy them. This
		// choice adds \Theta(2n) to the complexity.
		for (int i = 0; i < n; i++) {
			xReal[i] = inputReal[i];
			xImag[i] = inputImag[i];
		}
		
		// First phase - calculation
		int k = 0;
		for (int l = 1; l <= nu; l++) {
			while (k < n) {
				for (int i = 1; i <= n2; i++) {
					p = bitreverseReference(k >> nu1, nu);
					// direct FFT or inverse FFT
					 arg = constant * p / n;
					 c = Math.cos(arg);
					 s = Math.sin(arg);
					 tReal = xReal[k + n2] * c + xImag[k + n2] * s;
					 tImag = xImag[k + n2] * c - xReal[k + n2] * s;
					 xReal[k + n2] = xReal[k] - tReal;
					 xImag[k + n2] = xImag[k] - tImag;
					 xReal[k] += tReal;
					 xImag[k] += tImag;
					 k++;
				}
				k += n2;
			}
			k = 0;
			nu1--;
			n2 /= 2;
		}
		
		// Second phase - recombination
		k = 0;
		int r;
		while (k < n) {
			r = bitreverseReference(k, nu);
			if (r > k) {
				tReal = xReal[k];
				tImag = xImag[k];
				xReal[k] = xReal[r];
				xImag[k] = xImag[r];
				xReal[r] = tReal;
				xImag[r] = tImag;
			}
			k++;
		}
			
			
		double [] magArray = new double[xReal.length];
		for (int i = 0; i < magArray.length; i += 1) {
			magArray[i] = Math.sqrt(xReal[i] * xReal[i] + xImag[i] * xImag[i]);
			//System.out.println(magArray[i]);
				
		}	
		return magArray;
	}
	private static int bitreverseReference(int j, int nu) {
		int j2;
		int j1 = j;
		int k = 0;
		for (int i = 1; i <= nu; i++) {
			j2 = j1 / 2;
			k = 2 * k + j1 - 2 * j2;
			j1 = j2;
		}
		return k;
	}
	public static double[] dct(double[] x){

		int N = x.length;
		double[] y = new double [x.length];
		//Outer loop interates on frequency values.
		for(int k=0; k < N;k++){
			double sum = 0.0;
			//Inner loop iterates on time-series points.
			for(int n=0; n < N; n++){
				double arg = Math.PI*k*(2.0*n+1)/(2*N);
				double cosine = Math.cos(arg);
				double product = x[n]*cosine;
				sum += product;
			}

			double alpha;
			if(k == 0){
				alpha = 1.0/Math.sqrt(2);
			}else{
				alpha = 1;
			}
			y[k] = sum*alpha*Math.sqrt(2.0/N);
		}
		return y;
	}
	public static double[][] readSVMpara(File file) throws IOException {
		BufferedReader BRin= new BufferedReader(new FileReader(file));
		InputStream is = new BufferedInputStream(new FileInputStream(file));
		
    		double[][] matrix = null;
        String line;
        int row = 0;
        int size = 0;
        
       
        byte[] c = new byte[1024];
        int count = 0;
        int readChars = 0;
        boolean empty = true;
        while ((readChars = is.read(c)) != -1) {
            empty = false;
            for (int i = 0; i < readChars; ++i) {
                if (c[i] == '\n') {
                    ++count;
                }
            }
        }
        is.close();
        
        
        int vsize = count;

        while ((line = BRin.readLine()) != null) {
            String[] vals = line.trim().split("\\s+");
            // Lazy instantiation.
            if (matrix == null) {
                size = vals.length;
                matrix = new double[vsize][size];
            }
            
            for (int col = 0; col < size; col++) {
                matrix[row][col] = Double.parseDouble(vals[col]);
            }

            row++;
        }
        
        BRin.close();
        return matrix;
	}
	public static double[][] genPowSpec(double[] filSig, int winlen, int winstep){
       
    		int numofWindows = (filSig.length - winlen) / winstep + 1;
        int numofElements = winlen/2 + 1;
        double magSpec [] = new double[winlen];
        double powSpec [][] =  new double [numofWindows][numofElements];
        int i = 0;
        int startIndex = 0;
	    int stopIndex = winlen;
        
        while (stopIndex < filSig.length) {
	        double[] realPart = Arrays.copyOfRange(filSig, startIndex, stopIndex);
	        double[] imagPart = new double[realPart.length];
	        magSpec = fft(realPart, imagPart);
	        	       
	        for (int j = 0; j < numofElements; j++) {
	        		powSpec[i][j] = magSpec[j] * magSpec[j] / winlen;

	        }
	        startIndex += winstep;
	        stopIndex += winstep;
	        i += 1;
        }
    		return powSpec;	
    }
	public static double[] genEnergy(double[][] powSpec, int winlen, int winstep){
    			
    		double energy [] = new double[powSpec.length];
    		for (int i = 0; i < energy.length; i++) {   
	        energy[i] = 0;
	        for (int j = 0; j < powSpec[0].length; j++) {
	        		energy[i] += powSpec[i][j];
	        		
	        }
	        if (energy[i] == 0)
        		energy[i] = Double.MIN_VALUE;
        
    		}
    		return energy;
    }
	public static double[][] genMelFilterBank_T(int nfilt, int winlen, double sampleRate, double lowFre, double highFre){
    	//Compute a Mel-filterbank.
       
        int nfft = winlen;
        double [] melpoints = new double [nfilt + 2];
        double melLowFre = 2595 * Math.log10((1+(lowFre/700.)));
        double melHighFre = 2595 * Math.log10((1+(highFre/700.)));
        
        for (int b = 0; b < melpoints.length ; b++) {
 	   		melpoints[b] = melHighFre/(nfilt + 1) * b;
 	   		//System.out.println(melpoints[b]);
        }
        
        double[] melpointshz = new double [melpoints.length];
        for (int b = 0; b < melpoints.length ; b++) {
 	   		melpointshz[b] = 700*(Math.pow(10,melpoints[b]/2595.0)-1);
 	   		//System.out.println(melpointshz[b]);
        }
       
        double [] bin =  new double [melpoints.length];
        for (int b = 0; b < melpoints.length ; b++) {
 	   		bin[b] = (nfft+1)*melpointshz[b]/sampleRate;
 	   		bin[b] = (long) bin[b];
 	   		//System.out.println(bin[b]);
        }
        double[][]fbank = new double [nfilt][(nfft/2) + 1];
       
        for (int row = 0; row < nfilt; row++) { //nfilt
     	  		int start = (int) bin[row];
     	  		int finish = (int) bin[row+1];
     	  		for (int column = start;  column < finish; column++) {
     	  			fbank[row][column] = (column - bin[row]) / (bin[row+1]-bin[row]);
     	  			
     	  		}
  
     	  		start = (int) bin[row+1];
     	  		finish = (int) bin[row+2];
     	  		for (int column = start;  column < finish; column++) {
     	  			fbank[row][column] = (bin[row+2]-column) / (bin[row+2]-bin[row+1]);
     	  		}
     	  		
     	  		
     	  		
        	}
   // Transpose fbank     	
        	int fbank_T_height = fbank[0].length;
        	int fbank_T_width = fbank.length;	
        	double[][] fbank_T = new double [fbank_T_height][fbank_T_width];
        	
        	for (int row = 0; row <  fbank_T_height; row++){
        		for (int column = 0; column < fbank_T_width; column++){
        			fbank_T[row][column] = fbank[column][row];
        		}
        	}
        	
        	return fbank_T;
    }
	public static double [][] genPowSpecDotFilterBank(double [][] powSpec, double [][] fbank_T){
	    	double[][]fbfeat = new double [powSpec.length][fbank_T[0].length];  	
	    	double sum;
	    	
	  	for (int row = 0; row <  fbfeat.length; row++){
	   		for (int column = 0; column < fbfeat[0].length; column++){
	   			sum = 0;
	   			for (int i = 0; i < powSpec[0].length; i++) {
	   	            sum = sum + powSpec[row][i]* fbank_T[i][column];	
	   			}
	   			if (sum == 0)
	   				sum = Double.MIN_VALUE;	   			
	   			fbfeat[row][column] = Math.log(sum);

	   		}
	  	}
	  	return fbfeat;
    }
	public static double [][] genMFCC(double[][] DCT, double [] energy, int L){
    		int ncoeff = DCT[0].length;
      	int nframes = DCT.length;
      	int [] n = new int[ncoeff];
      	double [] lift = new double[ncoeff];

      	int i = 0;
      	while (i < ncoeff) {
       		n[i] = i;
       		lift[i] = 1 + (L/2.) * (Math.sin(Math.PI* n[i] / L));
      		i = i+ 1;
      	}
      	
      	for(i = 0; i < DCT.length; i++) {
      		for(int j = 0; j < DCT[0].length; j++) {
      		        DCT[i][j] = DCT[i][j] * lift[j];
      		       
      		 }
      		DCT[i][0] = Math.log(energy[i]);
      	}
      	
      	
      	for(i = 0; i < DCT.length; i++) {
      		for(int j = 0; j < DCT[0].length; j++) {
      		        if (Double.isNaN(DCT[i][j])) {
      		        		DCT[i][j] = Double.MIN_VALUE;
      		        }
      		        
      		        if (DCT[i][j] > 100.0) {
      		        		DCT[i][j] = 100.0;
      		        }else if (DCT[i][j] < -100.0){
      		        		DCT[i][j] = 100.0;
      		        }      
      		 }    		
      	}
      	
      	double [][] mfcc = DCT;
      	return mfcc;
    }
	public static double[][] nrm_datScaling(double [][] mfcc, double [][] scalefactors, double [][] shifts){ 
		for(int i = 0; i < mfcc.length; i++) {
	  		for(int j = 0; j < mfcc[0].length; j++) {
	  			mfcc[i][j] = mfcc[i][j]/scalefactors[j][0] - shifts[j][0];
	  		}
		}
		return mfcc;
	}
	public static double [] calcPredProb(double []dec, double[][]A, double[][]B){
	
		
	    	double[] pred_prob = new double [dec.length];
	    for (int i=0; i < pred_prob.length; i++) {
	            double fApB = dec[i]*A[0][0] + B[0][0];
	            if (fApB >= 0) {
	                pred_prob[i] = (Math.exp(-1.0 * fApB))/(1.0+(Math.exp(-fApB)));
	            }else {
	                pred_prob[i] = 1.0/(1+(Math.exp(fApB)));
	            }    
	    	}  	
		return pred_prob;    
		}
	public static double[][] calc_rbfs(double [][] mfcc, double [][] SV, double [][] RBF_gamma) {
		double [][] rbfs = new double[mfcc.length][SV.length];
        	for(int z = 0; z < mfcc.length; z++) {
        		for(int i = 0; i < SV.length; i++) {
	        		double sum = 0.0;
		        	for (int j = 0; j < mfcc[0].length;j++) {
		        		sum += (mfcc[z][j] - SV[i][j]) * (mfcc[z][j] - SV[i][j]);	
		        	}
		        	rbfs[z][i] = Math.exp(-1 * sum * RBF_gamma[0][0]);	
        		}       		
        	}
        	return rbfs;
	}
	public static double [] calc_dec(double[][] rbfs, double[][] dual_coef, double[][] intercept) {
		double [] dec = new double[rbfs.length];
		for(int z = 0; z < rbfs.length; z++) {
	    			dec[z] = 0;
		        	for (int j = 0; j < rbfs[0].length;j++) {
		        		
		        		dec[z] += rbfs[z][j] * dual_coef[0][j] ;			
		        	}
		        	dec[z] = dec[z] + intercept[0][0];
	    	}
		return dec;
	}
	
	
	
	
	public static void main(String[] args) throws IOException, UnsupportedAudioFileException, WavFileException {
		long startTime_INCLOAD = System.nanoTime();
//input parameters
		double filterCoeff = 0.97;
		int winlen = 512;
	    int winstep = winlen/2;
	    int nfilt = 26;
	    double sampleRate = 8000;
	    double lowFre = 0.0;
	    double highFre = 4000;
	    int L = 22;
	    int i;
//load SVM parameters
	    	double [][] SV= readSVMpara(new File("/Users/tsunhenry/Downloads/MozzLive-master 2/SV.txt"));
	    	System.out.println("SV = " + SV.length + "x" + SV[0].length);
	    double [][] dual_coef= readSVMpara(new File("/Users/tsunhenry/Downloads/MozzLive-master 2/dual_coef.txt"));
	    	System.out.println("dual_coef = " + dual_coef.length + "x" + dual_coef[0].length);
	    	double [][] A= readSVMpara(new File("/Users/tsunhenry/Downloads/MozzLive-master 2/A.txt"));
	    	System.out.println("A = " + A.length + "x" + A[0].length + A[0][0]);
	    	double [][] B= readSVMpara(new File("/Users/tsunhenry/Downloads/MozzLive-master 2/B.txt"));
	    	System.out.println("B = " + B.length + "x" + B[0].length + B[0][0]);
	    	double [][] intercept= readSVMpara(new File("/Users/tsunhenry/Downloads/MozzLive-master 2/intercept.txt"));
	    	System.out.println("intercept = " + intercept.length + "x" + intercept[0].length + " : " + intercept[0][0]);
	    double [][] RBF_gamma= readSVMpara(new File("/Users/tsunhenry/Downloads/MozzLive-master 2/RBF_gamma.txt"));
	    	System.out.println("RBF_gamma = " + RBF_gamma.length + "x" + RBF_gamma[0].length + RBF_gamma[0][0]);
	    	double [][] shifts= readSVMpara(new File("/Users/tsunhenry/Downloads/MozzLive-master 2/shifts.txt"));
	 	System.out.println("shifts = " + shifts.length + "x" + shifts[0].length);
	 	double [][] scalefactors= readSVMpara(new File("/Users/tsunhenry/Downloads/MozzLive-master 2/scalefactors.txt"));
	 	System.out.println("scalefactors = " + scalefactors.length + "x" + scalefactors[0].length);   
// Create textfile to write magnitude fft results 
        
        BufferedWriter probOut = new BufferedWriter(new FileWriter("pred_prob.txt"));
        
		
// ---------------------------------------Start-------------------------------------
        long startTime = System.nanoTime();
//1) Read the input data, convert raw wavefile to array
        double[] inputData = readFully(new File("0032_norm.wav"));
//2) Filter Signal - equivalent to sig.proc
        double[] filSig = filterSignal(inputData, filterCoeff);
//3) Generate PowSpec and Energy
        double powSpec [][] =  genPowSpec(filSig, winlen, winstep);
        double energy [] = genEnergy(powSpec, winlen, winstep);
//4) Compute a transposed Mel-filterbank.	
       	double[][] fbank_T = genMelFilterBank_T(nfilt, winlen, sampleRate, lowFre, highFre);	
//5) Dot product between Mel-filterbank and powSpec, and take log
    	  	double[][]fbfeat = genPowSpecDotFilterBank(powSpec, fbank_T); 	
//6) DCT	
      	double [][] temp_DCT = new double [fbfeat.length][fbfeat[0].length];
      	for (i = 0; i < temp_DCT.length; i++ ) {
      			temp_DCT[i] = dct(fbfeat[i]);
      	}
      	
      	double [][] DCT = new double [temp_DCT.length][temp_DCT[0].length/2];
      	for (i = 0; i < DCT.length; i++){
      		for (int j = 0; j < DCT[0].length; j++){
      			DCT[i][j] = temp_DCT[i][j];
      		}
      	}
//7) MFCC production 
      	double [][] mfcc = genMFCC(DCT, energy, L);	
      	//Save MFCC to file "feature.txt"
      	saveMatrixAsText("feature.txt", mfcc);
//8)nrm_dat scaling
        	mfcc = nrm_datScaling(mfcc, scalefactors, shifts);   	
//9) Calculate rbfs and dec  
		double [][] rbfs = calc_rbfs(mfcc, SV, RBF_gamma);
	    	double [] dec = calc_dec(rbfs, dual_coef, intercept);
	    	//Save dec to file "dec.txt"
        	saveArrayAsText("dec.txt", dec);	
        	
//10) Calculate pred_prob 
        	double[] pred_prob = calcPredProb(dec, A, B);
        	//Save pred_prob to file "pred_prob.txt"
        saveArrayAsText("pred_prob.txt", pred_prob);	    
//----------------------------------------end---------------------------------------------        
        long endTime = System.nanoTime();
        System.out.println("done");
    	  	System.out.println("Took "+((endTime - startTime_INCLOAD)*1e9) + "s including loading parameters"); 
    	  	System.out.println("Took "+((endTime - startTime)*1e9) + "s to calculate"); 
       	} 
}
