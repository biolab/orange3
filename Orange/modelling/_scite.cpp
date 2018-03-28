/*
 * findBestTrees_noR.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: jahnka
 */

#include <stdbool.h>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <string>
#include <float.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <cmath>
#include <queue>
#include <random>
#include "limits.h"


#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif // _WIN32

using namespace std;



/* Headers */

// matrices.h
int** sumMatrices(int** first, int** second, int n, int m);
double getMaxEntry(double* array, int n);
int ** transposeMatrix(int** matrix, int n, int m);
void addToMatrix(int** first, int** second, int n, int m);
double** allocate_doubleMatrix(int n, int m);
int** allocate_intMatrix(int n, int m);
bool** allocate_boolMatrix(int n, int m);
double** init_doubleMatrix(int n, int m, double value);
int* init_intArray(int n, int value);
bool* init_boolArray(int n, bool value);
int** init_intMatrix(int n, int m, int value);
bool** init_boolMatrix(int n, int m, bool value);
double* init_doubleArray(int n, double value);
void reset_intMatrix(int** matrix, int n, int m, int value);
void free_boolMatrix(bool** matrix);
void free_intMatrix(int** matrix);
void free_doubleMatrix(double** matrix);
bool** deepCopy_boolMatrix(bool** matrix, int n, int m);
int** deepCopy_intMatrix(int** matrix, int n, int m);
int* deepCopy_intArray(int* array, int n);
double* deepCopy_doubleArray(double* array, int n);
double** deepCopy_doubleMatrix(double** matrix, int n, int m);
void print_boolMatrix(bool** array, int n, int m);
void print_doubleMatrix(double** matrix, int n, int m);
void print_intMatrix(int** matrix, int n, int m, char del);
void print_intArray(int* array, int n);
int* ancMatrixToParVector(bool** anc, int n);
bool identical_boolMatrices(bool** first, bool** second, int n, int m);
void delete_3D_intMatrix(int*** matrix, int n);


// treelist.h
struct treeBeta
{
	int* tree;
    double beta;
};
void updateTreeList(std::vector<struct treeBeta>& bestTrees, int* currTreeParentVec, int n, double currScore, double bestScore, double beta);
void resetTreeList(std::vector<struct treeBeta>& bestTrees, int* newBestTree, int n, double beta);
void emptyVectorFast(std::vector<struct treeBeta>& optimalTrees, int n);
void emptyTreeList(std::vector<int*>& optimalTrees, int n);
struct treeBeta createNewTreeListElement(int* tree, int n, double beta);
bool isDuplicateTreeFast(std::vector<struct treeBeta> &optimalTrees, int* newTree, int n);

// trees.h
std::vector<int> getDescendants(bool** ancMatrix, int node, int n);
std::vector<int> getNonDescendants(bool**& ancMatrix, int node, int n);
int countBranches(int* parents, int length);
vector<vector<int> > getChildListFromParentVector(int* parents, int n);
void deleteChildLists(vector<vector<int> > &childLists);
string getNewickCode(vector<vector<int> > list, int root);
int* prueferCode2parentVector(int* code, int codeLength);
int* getBreadthFirstTraversal(int* parent, int n);
bool** parentVector2ancMatrix(int* parent, int n);
int* getRandParentVec(int n);
bool* getInitialQueue(int* code, int codeLength);
int* getLastOcc(int* code, int codeLength);
int getNextInQueue(bool* queue, int pos, int length);
void updateQueue(int node, bool* queue, int next);
int updateQueueCutter(int node, bool* queue, int next);
int* starTreeVec(int n);
bool** starTreeMatrix(int n);
int* reverse(int* array, int length);

// output.h
double binTreeRootScore(int** obsMutProfiles, int mut, int m, double ** logScores);
int getHighestOptPlacement(int** obsMutProfiles, int mut, int m, double ** logScores, bool** ancMatrix);
int* getHighestOptPlacementVector(int** obsMutProfiles, int n, int m, double ** logScores, bool** ancMatrix);
std::vector<std::string> getBinTreeNodeLabels(int nodeCount, int* optPlacements, int n, std::vector<std::string> geneNames);
int getLcaWithLabel(int node, int* parent, std::vector<std::string> label, int nodeCount);
std::string getGraphVizBinTree(int* parents, int nodeCount, int m, std::vector<std::string> label);
std::string getMutTreeGraphViz(std::vector<std::string> label, int nodeCount, int m, int* parent);
void writeToFile(std::string content, std::string fileName);
std::string getGraphVizFileContentNumbers(int* parents, int n);
std::string getGraphVizFileContentNames(int* parents, int n, std::vector<std::string> geneNames, bool attachSamples, bool** ancMatrix, int m, double** logScores, int** dataMatrix);
std::string getBestAttachmentString(bool ** ancMatrix, int n, int m, double** logScores, int** dataMatrix, std::vector<std::string> geneNames);
bool** attachmentPoints(bool ** ancMatrix, int n, int m, double** logScores, int** dataMatrix);
void printParentVectors(std::vector<bool**> optimalTrees, int n, int m, double** logScores, int** dataMatrix);
void printGraphVizFile(int* parents, int n);
void printSampleTrees(std::vector<int*> list, int n, std::string fileName);
void printScoreKimSimonTree(int n, int m, double** logScores, int** dataMatrix, char scoreType);

// mcmcTreeMove.h
int* proposeNewTree(std::vector<double> moveProbs, int n, bool** currTreeAncMatrix, int* currTreeParentVec, double& nbhcorrection);
int choseParent(std::vector<int> &possibleParents, int root);
int* getNewParentVecFast(int* currTreeParentVec, int nodeToMove, int newParent, int n);
int* getNewParentVec_SwapFast(int* currTreeParentVec, int first, int second, int n);
int* reorderToStartWithDescendant(int* nodestoswap, bool** currTreeAncMatrix);
int* getNewParentVec_Swap(int* currTreeParentVec, int first, int second, int n, int* propTreeParVec);
bool** getNewAncMatrix_Swap(bool** currTreeAncMatrix, int first, int second, int n, bool** propTreeAncMatrix);
int* getNewParentVec(int* currTreeParentVec, int nodeToMove, int newParent, int n, int *propTreeParVec);
bool** getNewAncMatrix(bool** currTreeAncMatrix, int newParent, std::vector<int> descendants, std::vector<int> possibleParents, int n, bool** propTreeAncMatrix);

//mcmcBinTreeMove.h
int* proposeNextBinTree(std::vector<double> moveProbs, int m, int* currTreeParVec, bool** currTreeAncMatrix);
int pickNodeToMove(int* currTreeParentVec, int parentVectorLength);
int getSibling(int v, int* currTreeParVec, std::vector<std::vector<int> > &childLists);

// mcmc.h
string runMCMCbeta(vector<struct treeBeta>& bestTrees, double* errorRates, int noOfReps, int noOfLoops, double gamma1, vector<double> moveProbs, int n, int m, int** dataMatrix, char scoreType, int* trueParentVec, int step, bool sample, double chi, double priorSd, bool useTreeList, char treeType);
double logBetaPDF(double x, double bpriora, double bpriorb);
double proposeNewBeta(double currBeta, double jumpSd);
double sampleNormal(double mean, double sd);
string sampleFromPosterior(double currTreeLogScore, int n, int* currTreeParentVec, double betaProb, double currBeta, double currScore);
int updateMinDistToTrueTree(int* trueParentVec, int* currTreeParentVec, int length, int minDistToTrueTree, int currScore, int bestScore);
int getSimpleDistance(int* trueVector, int* predVector, int length);

// random.h
void initRand();
bool changeBeta(double prob);
int sampleRandomMove(std::vector<double> prob);
int* sampleTwoElementsWithoutReplacement(int n);
int pickRandomNumber(int n);
double sample_0_1();
int* getRandTreeCode(int n);
bool samplingByProb(double prob);
int* getRandomBinaryTree(int m);

// scoreBinTree.h
double* getBinSubtreeScore(bool state, int* bft, std::vector<std::vector<int> > &childLists, int mut, int nodeCount, int m, int** obsMutProfiles, double ** logScores);
double getBinTreeMutScore(int* bft, std::vector<std::vector<int> > &childLists, int mut, int nodeCount, int m, int** obsMutProfiles, double ** logScores);
double getBinTreeScore(int** obsMutProfiles, int n, int m, double ** logScores, int* parent);

// scoreTree.h
double scoreTree(int n, int m, double** logScores, int** dataMatrix, char type, int* parentVector, double bestScore);
double scoreTreeFast(int n, int m, double** logScores, int** dataMatrix, char type, int* parentVector);
double maxScoreTreeFast(int n, int m, double** logScores, int** dataMatrix, int* parent, int* bft);
double sumScoreTreeFast(int n, int m, double** logScores, int** dataMatrix, int* parent, int* bft);
double* getAttachmentScoresFast(int*parent, int n, double** logScores, int* dataVector, int*bft);
double rootAttachementScore(int n, double** logScores, int* mutationVector);
double scoreTreeAccurate(int n, int m, double** logScores, int** dataMatrix, char type, int* parentVector);
double maxScoreTreeAccurate(int n, int m, double** logScores, int** dataMatrix, int* parent, int* bft);
double sumScoreTreeAccurate(int n, int m, double** logScores, int** dataMatrix, int* parent, int* bft);
int** getBestAttachmentScoreAccurate(int** scoreMatrix, int* parent, int n, double** logScores, int* dataVector, int* bft);
int*** getAttachmentMatrices(int* parent, int n, int* dataVector, int* bft);
double getTrueScore(int** matrix, double** logScores);
double getSumAttachmentScoreAccurate(int* parent, int n, double** logScores, int* dataVector, int* bft);
double** getLogScores(double FD, double AD1, double AD2, double CC);
void updateLogScores(double** logScores, double newAD);
double** getScores(double FD, double AD1, double AD2, double CC);
double* getTrueScores(int*** matrix, int n, double** logScores);
void printLogScores(double** logScores);

// findBestTrees.h
int** getDataMatrix(int n, int m, string fileName);
double* getErrorRatesArray(double fd, double ad1, double ad2, double cc);
int readParameters(int argc, char* argv[]);
string getOutputFilePrefix(string fileName, string outFile);
string getFileName(string prefix, string ending);
string getFileName2(int i, string prefix, string ending, char scoreType);
vector<string> getGeneNames(string fileName, int nOrig);
vector<double> setMoveProbs();
int* getParentVectorFromGVfile(string fileName, int n);
int getMinDist(int* trueVector, std::vector<bool**> optimalTrees, int n);
void printGeneFrequencies(int** dataMatrix, int n, int m, vector<string> geneNames);

// Function headers provided by findBestTrees.cpp
void printGeneFrequencies(int** dataMatrix, int n, int m, vector<string> geneNames);
int* getParentVectorFromGVfile(string fileName, int n);
int getMinDist(int* trueVector, std::vector<bool**> optimalTrees, int n);
string getOutputFilePrefix(string fileName, string outFile);
string getFileName(string prefix, string ending);
string getFileName2(int i, string prefix, string ending, char scoreType);
int readParameters(int argc, char* argv[]);
vector<double> setMoveProbs();
int** getDataMatrix(int n, int m, string fileName);
vector<string> getGeneNames(string fileName, int nOrig);
double* getErrorRatesArray(double fd, double ad1, double ad2, double cc);



/*****    basic functions on 1D and 2D arrays   *****/

double getMaxEntry(double* array, int n){
	double maxEntry = -DBL_MAX;
	for(int i=0; i<n; i++){
		maxEntry = max(maxEntry, array[i]);
	}
	return maxEntry;
}

int** sumMatrices(int** first, int** second, int n, int m){
	for(int i=0; i<n; i++){
		for(int j=0; j<m; j++){
			first[i][j] += second[i][j];
		}
	}
	return first;
}

int ** transposeMatrix(int** matrix, int n, int m){
	int ** transposed = allocate_intMatrix(m, n);
	for(int i=0; i<m; i++){
		for(int j=0; j<n;j++){
			transposed[i][j] = matrix[j][i];
		}
	}
	return transposed;
}

void addToMatrix(int** first, int** second, int n, int m){
	for(int i=0; i<n; i++){
		for(int j=0; j<m; j++){
			first[i][j] += second[i][j];
		}
	}
}

int* ancMatrixToParVector(bool** anc, int n){
	int* parVec = new int[n];
	for(int i=0; i<n; i++){
		parVec[i] = n;
	}
	for(int i=0; i<n; i++){
		for(int k=0; k<n; k++){
			if(k!=i && anc[k][i]==true){  // k is true ancestor of i
				bool cand = true;
				for(int l=0; l<n; l++){
					if(l!=i && l!=k && anc[l][i]==true && anc[k][l]==true){   // k is ancestor of l, and l is ancestor of i
						cand = false;                                        // k is no longer candidate for being parent of i
						break;
					}
				}
				if(cand==true){           // no ancestor of i is descendant of k -> k is parent of i
						parVec[i] = k;
				}
			}

		}
	}
	return parVec;
}

/*   allocation  */
double** allocate_doubleMatrix(int n, int m){

    double** matrix = new double*[n];
    matrix[0] = new double[n*m];
	  for (int i=1; i<n; ++i)
    {
        matrix[i] = matrix[i-1] + m;
    }
    return matrix;
}

int** allocate_intMatrix(int n, int m){

    int** matrix = new int*[n];
    matrix[0] = new int[n*m];
    for (int i=1; i<n; ++i)
    {
        matrix[i] = matrix[i-1] + m;
    }
    return matrix;
}

bool** allocate_boolMatrix(int n, int m){

    bool** matrix = new bool*[n];
    matrix[0] = new bool[n*m];
    for (int i=1; i<n; ++i)
    {
        matrix[i] = matrix[i-1] + m;
    }
    return matrix;
}


/*  initialization  */

int* init_intArray(int n, int value){
	int* array = new int[n];
	for(int i=0; i<n; i++){
		array[i] = value;
	}
	return array;
}

double* init_doubleArray(int n, double value){
	double* array = new double[n];
	for(int i=0; i<n; i++){
		array[i] = value;
	}
	return array;
}

bool* init_boolArray(int n, bool value){
	bool* array = new bool[n];
	for(int i=0; i<n; i++){
		array[i] = value;
	}
	return array;
}

double** init_doubleMatrix(int n, int m, double value){

	  double** matrix = allocate_doubleMatrix(n, m);     // allocate

    for (int i=0; i<n; ++i)             // initialize
    {
         for (int j=0; j<m; ++j)
      {
        	matrix[i][j] = value;
    	}
    }
    return matrix;
}

int** init_intMatrix(int n, int m, int value){

    int** matrix = allocate_intMatrix(n, m);  // allocate

    for (int i=0; i<n; ++i)            // initialize
    {
         for (int j=0; j<m; ++j)
      {
        	matrix[i][j] = value;
    	}
    }
    return matrix;
}

void reset_intMatrix(int** matrix, int n, int m, int value){

    for (int i=0; i<n; ++i)            // initialize
    {
         for (int j=0; j<m; ++j)
      {
        	matrix[i][j] = value;
    	}
    }
}


bool** init_boolMatrix(int n, int m, bool value){

    bool** matrix = allocate_boolMatrix(n, m);     // allocate

    for (int i=0; i<n; ++i)             // initialize
    {
         for (int j=0; j<m; ++j)
      {
        	matrix[i][j] = value;
    	}
    }
    return matrix;
}


/*  deallocation  */

void delete_3D_intMatrix(int*** matrix, int n){
	for(int i=0; i<n; i++){
			delete [] matrix[i][0];
			delete [] matrix[i];
		}
		delete [] matrix;
}


void free_boolMatrix(bool** matrix){
    delete [] matrix[0];
    delete [] matrix;
}

void free_intMatrix(int** matrix){
    delete [] matrix[0];
    delete [] matrix;
}

void free_doubleMatrix(double** matrix){
    delete [] matrix[0];
    delete [] matrix;
}


/*  deep copying  */

bool** deepCopy_boolMatrix(bool** matrix, int n, int m){
    bool** deepCopy = init_boolMatrix(n,m, false);
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<m; j++)
        {
    	      deepCopy[i][j] = matrix[i][j];
	      }
    }
    return deepCopy;
}

int** deepCopy_intMatrix(int** matrix, int n, int m){
    int** deepCopy = init_intMatrix(n,m, -1);
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<m; j++)
        {
    	      deepCopy[i][j] = matrix[i][j];
	      }
    }
    return deepCopy;
}

double** deepCopy_doubleMatrix(double** matrix, int n, int m){
    double** deepCopy = init_doubleMatrix(n,m, -1);
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<m; j++)
        {
    	      deepCopy[i][j] = matrix[i][j];
	      }
    }
    return deepCopy;
}

int* deepCopy_intArray(int* array, int n){
	  int* deepCopy = new int[n];
	  for (int i=0; i<n; ++i)
    {
        deepCopy[i] = array[i];
    }
    return deepCopy;
}

double* deepCopy_doubleArray(double* array, int n){
	double* deepCopy = new double[n];
	for (int i=0; i<n; ++i)
    {
        deepCopy[i] = array[i];
    }
    return deepCopy;
}

bool identical_boolMatrices(bool** first, bool** second, int n, int m){
	for(int i=0; i<n; i++){
		for(int j=0; j<m; j++){
			//cout << "[" << i << "," << j << "] ";
			if(first[i][j] != second[i][j]){
				// cout << "matrices differ!!!!!!!!!!!!!!!!\n";
				getchar();
				return false;
			}
		}
		//cout << "\n";
	}
	return true;
}

/*  printing  */

void print_boolMatrix(bool** array, int n, int m){
	  for(int i=0; i<n; i++){
		    for(int j=0; j<m; j++){
			      // cout << array[i][j] << " ";
		    }
		    // cout << "\n";
	  }
}

void print_doubleMatrix(double** matrix, int n, int m){
	  for(int i=0; i<n; i++){
  		  for(int j=0; j<m; j++){
  			    // cout << matrix[i][j] << "\t";
  		  }
  		  // cout << "\n";
  	}
}


void print_intArray(int* array, int n){
	for (int i=0; i<n; ++i)
    {
       // cout << array[i] << " ";
    }
    // cout << "\n";
}

void print_intMatrix(int** matrix, int n, int m, char del){
	  for(int i=0; i<n; i++){
  		  for(int j=0; j<m; j++){
  			    // cout << matrix[i][j] <<  del;
  		  }
  		  // cout << "\n";
  	}
}



void updateTreeList(vector<struct treeBeta>& bestTrees, int* currTreeParentVec, int n, double currScore, double bestScore, double beta){

	if(currScore > bestScore){
		//cout << "tree list of size " << bestTrees.size() << " emptied\n";
		resetTreeList(bestTrees, currTreeParentVec, n, beta);                              // empty the list of best trees and insert current tree

	}
	else if (currScore == bestScore){
		if(!isDuplicateTreeFast(bestTrees, currTreeParentVec, n)){               // if the same tree was not previously found
			treeBeta newElem = createNewTreeListElement(currTreeParentVec, n, beta);
			bestTrees.push_back(newElem);        // add it to list
		}
	}
}


/* removes all elements from the vector and inserts the new best tree */
void resetTreeList(vector<struct treeBeta>& bestTrees, int* newBestTree, int n, double beta){
	emptyVectorFast(bestTrees, n);                                         // empty the list of best trees
	treeBeta newElem = createNewTreeListElement(newBestTree, n, beta);
	bestTrees.push_back(newElem);                // current tree is now the only best tree
}


/* removes all elements from the vector */
void emptyVectorFast(std::vector<struct treeBeta>& optimalTrees, int n){
    for(int i=0; i<optimalTrees.size(); i++){
    	delete [] optimalTrees[i].tree;
	}
    optimalTrees.clear();
}

/* removes all elements from the vector */
void emptyTreeList(std::vector<int*>& optimalTrees, int n){
    for(int i=0; i<optimalTrees.size(); i++){
    	delete [] optimalTrees[i];
	}
    optimalTrees.clear();
}

/* creates a new tree/beta combination */
struct treeBeta createNewTreeListElement(int* tree, int n, double beta){
	treeBeta newElem;
	newElem.tree = deepCopy_intArray(tree, n);
	newElem.beta = beta;
	return newElem;
}

/* returns true if the same tree was found before */
bool isDuplicateTreeFast(std::vector<struct treeBeta> &optimalTrees, int* newTree, int n){
    for(int k=0; k<optimalTrees.size(); k++){
      bool same = true;
      for(int i=0; i<n; i++){
    	  if(newTree[i] != optimalTrees[k].tree[i]){
              same = false;
              break;
          }
      }
      if(same == true){
        return true;
      }
    }
    return false;
}


/* returns all nodes that are descendants of the given node */
/* note: ancMatrix is 1 at [i,j] if i is an ancestor of j in the tree */
std::vector<int> getDescendants(bool** ancMatrix, int node, int n){
  std::vector<int> descendants;
  for(int i=0; i<n; i++){
  	if(ancMatrix[node][i]==true){
			descendants.push_back(i);
		}
	}
	return descendants;
}

/* returns all nodes that are not descendants of the given node */
/* i.e. ancestors and nodes in a different branch of the tree   */
/* note: ancMatrix is 0 at [i,j] if i is not an ancestor of j in the tree */
std::vector<int> getNonDescendants(bool**& ancMatrix, int node, int n){
	std::vector<int> ancestors;
	for(int i=0; i<n; i++){
		if(ancMatrix[node][i]==false){
			ancestors.push_back(i);
		}
	}
	return ancestors;
}

/* counts the number of branches in a tree, this is the same as the number of leafs in the tree */
int countBranches(int* parents, int length){
	int count = 0;
	vector<vector<int> > childList = getChildListFromParentVector(parents, length);
	for(int i=0; i<childList.size(); i++){
		if(childList.at(i).size()==0){ count++; }
	}
	for(int i=0; i<childList.size(); i++){
		childList[i].clear();
	}
	childList.clear();
	return count;
}

/* converts a parent vector to the list of children */
vector<vector<int> > getChildListFromParentVector(int* parents, int n){

	vector<vector<int> > childList(n+1);
	for(int i=0; i<n; i++){
		childList.at(parents[i]).push_back(i);
	}
	return childList;
}

void deleteChildLists(vector<vector<int> > &childLists){
	for(int i=0; i<childLists.size(); i++){
		childLists[i].clear();
	}
	childLists.clear();
}

/* converts a tree given as lists of children to the Newick tree format */
/* Note: This works only if the recursion is started with the root node which is n+1 */
string getNewickCode(vector<vector<int> > list, int root){
	stringstream newick;
	vector<int> rootChilds = list.at(root);
	if(!rootChilds.empty()){
		newick << "(";
		bool first = true;
		for(int i=0; i<rootChilds.size(); i++){
			if(!first){
				newick << ",";
			}
			first = false;
			newick << getNewickCode(list, rootChilds.at(i));
		}
		newick << ")";
	}
	newick << root+1;
	return newick.str();
}



/*  computes a breadth first traversal of a tree from the parent vector  */
int* getBreadthFirstTraversal(int* parent, int n){

	vector<vector<int> > childLists = getChildListFromParentVector(parent, n);
	int* bft = new int[n+1];
	bft[0] = n;
	int k = 1;

	for(int i=0; i<n+1; i++){
		for(int j=0; j<childLists[bft[i]].size(); j++){
			bft[k++] = childLists[bft[i]][j];
		}
	}
	for(int i=0; i<childLists.size(); i++){
		childLists[i].clear();
	}
	childLists.clear();
	return bft;
}

int* reverse(int* array, int length){
	int temp;

	for (int i = 0; i < length/2; ++i) {
		temp = array[length-i-1];
		array[length-i-1] = array[i];
		array[i] = temp;
	}
	return array;
}



/* transforms a parent vector to an ancestor matrix*/
bool** parentVector2ancMatrix(int* parent, int n){
	bool** ancMatrix = init_boolMatrix(n, n, false);
	int root = n;
	for(int i=0; i<n; i++){
		int anc = i;
		int its =0;
		while(anc < root){                              // if the ancestor is the root node, it is not represented in the adjacency matrix
			if(parent[anc]<n){
				ancMatrix[parent[anc]][i] = true;
			}

			anc = parent[anc];
			its++;
		}
	}
	for(int i=0; i<n; i++){
		ancMatrix[i][i] = true;
	}
	return ancMatrix;
}

/* given a Pruefer code, compute the corresponding parent vector */
int* prueferCode2parentVector(int* code, int codeLength){
	int nodeCount = codeLength+1;
	int* parent = new int[nodeCount];
	//print_intArray(code, codeLength);
	int* lastOcc = getLastOcc(code, codeLength);    // node id -> index of last occ in code, -1 if no occurrence or if id=root
	bool* queue = getInitialQueue(code, codeLength);  // queue[node]=true if all children have been attached to this node, or if it is leaf
	int queueCutter = -1;    // this is used for a node that has been passed by the "queue" before all children have been attached
	int next = getNextInQueue(queue, 0, codeLength+1);

	for(int i=0; i<codeLength; i++){               // add new edge to tree from smallest node with all children attached to its parent
		if(queueCutter >=0){
			parent[queueCutter] = code[i];         // this node is queueCutter if the queue has already passed this node
			//cout << queueCutter << " -> " << code[i] << "\n";
			queueCutter = -1;
		}
		else{
			parent[next] = code[i];                               // use the next smallest node in the queue, otherwise
			//cout << next << " -> " << code[i] << "\n";
			next = getNextInQueue(queue, next+1, codeLength+1);     // find next smallest element in the queue
		}

		if(lastOcc[code[i]]==i){                               // an element is added to the queue, or we have a new queueCutter
			updateQueue(code[i], queue, next);
			queueCutter = updateQueueCutter(code[i], queue, next);
		}
	}
	if(queueCutter>=0){
		parent[queueCutter] = nodeCount;
		//cout << queueCutter << " -> " << nodeCount << "\n";
	}
	else{
		parent[next] = nodeCount;
		//cout << next << " -> " << nodeCount << "\n";
	}

	delete [] lastOcc;
	delete [] queue;
	//print_intArray(parent, codeLength+1);
	//getGraphVizFileContentNumbers(parent, codeLength+1);
	return parent;
}

bool* getInitialQueue(int* code, int codeLength){
	//cout << "code Length: " << codeLength << "\n";
	int queueLength = codeLength+2;
	//cout << "queueLength: " << queueLength << "\n";
	bool* queue = init_boolArray(queueLength, true);

	for(int i=0; i<codeLength; i++){
		queue[code[i]] = false;
	}
	return queue;
}


void updateQueue(int node, bool* queue, int next){

	if(node>=next){                //  add new node to queue
		queue[node] = true;
	}
}

int updateQueueCutter(int node, bool* queue, int next){
	if(node>=next){
		return -1;         // new node can be added to the queue
	}
	else{
		return node;         // new node needs to cut the queue, as it has already passed it
	}
}


int* getLastOcc(int* code, int codeLength){
	int* lastOcc = init_intArray(codeLength+2, -1);
	int root = codeLength+1;
	for(int i=0; i<codeLength; i++){
		if(code[i] != root){
			lastOcc[code[i]] = i;
		}
	}
	return lastOcc;
}

int getNextInQueue(bool* queue, int pos, int length){
	for(int i=pos; i<length; i++){
		if(queue[i]==true){
			return i;
		}
	}
	//cout << "No node left in queue. Possibly a cycle?";
	return length;
}

/* creates a random parent vector for nodes 0, .., n with node n as root*/
int* getRandParentVec(int n){
	int* randCode = getRandTreeCode(n);
	int* randParent = prueferCode2parentVector(randCode, n-1);
	delete [] randCode;
	return randParent;
}



/* creates the parent vector for a star tree with node n as center and 0,...,n-1 as leafs */
int* starTreeVec(int n){
	int* starTreeVec = new int[n];
	for(int i=0;i<n;i++){
		starTreeVec[i] = n;
	}
	return starTreeVec;
}

/* creates the ancestor matrix for the same tree */
bool** starTreeMatrix(int n){

  bool** starTreeMatrix = init_boolMatrix(n, n, false);
  for(int i=0;i<n;i++){
		starTreeMatrix[i][i] = true;
	}
	return starTreeMatrix;
}


/* Score contribution by a specific mutation when placed at the root, that means all samples should have it */
/* This is the same for all trees and can be precomputed */
double binTreeRootScore(int** obsMutProfiles, int mut, int m, double ** logScores){
	double score = 0.0;
	for(int sample=0; sample<m; sample++){
		score += logScores[obsMutProfiles[sample][mut]][1];
	}
	return score;
}

/* computes the best placement of a mutation, the highest one if multiple co-opt. placements exist*/
int getHighestOptPlacement(int** obsMutProfiles, int mut, int m, double ** logScores, bool** ancMatrix){

	int nodeCount = (2*m)-1;
	int bestPlacement = (2*m)-2;   // root
	double bestPlacementScore = binTreeRootScore(obsMutProfiles, mut, m, logScores);
	//cout << bestPlacementScore << " (root)\n";
	//print_boolMatrix(bool** array, int n, int m);
	for(int p=0; p<nodeCount-1; p++){                           // try all possible placements (nodes in the mutation tree)

		double score = 0.0;                   // score for placing mutation at a specific node
		for(int sample=0; sample<m; sample++){
			//cout << p << " " << sample << "\n";
			if(ancMatrix[p][sample] == 1){
				score += logScores[obsMutProfiles[sample][mut]][1]; // sample should have the mutation
			}
			else{
				score += logScores[obsMutProfiles[sample][mut]][0]; // sample should not have the mutation
			}
		}
		if(score > bestPlacementScore){
			bestPlacement = p;
			bestPlacementScore = score;
			//cout << bestPlacementScore << " (non-root)\n";
		}
		else if (score == bestPlacementScore && ancMatrix[p][bestPlacement] == true){
			bestPlacement = p;
		}
	}

	//if(bestPlacement == (2*m)-2){
	//	cout<< "best placed at root\n";
	//	getchar();
	//}
	return bestPlacement;
}

/* computes the best placement of a mutation, the highest one if multiple co-opt. placements exist*/
int* getHighestOptPlacementVector(int** obsMutProfiles, int n, int m, double ** logScores, bool** ancMatrix){
	int* bestPlacements = init_intArray(n, -1);
	for(int mut=0; mut<n; mut++){                                                               // for all mutations get
		bestPlacements[mut] = getHighestOptPlacement(obsMutProfiles, mut, m, logScores, ancMatrix);         // bestPlacementScore
	 }
	//print_intArray(bestPlacements, n);
	return bestPlacements;
}

vector<string> getBinTreeNodeLabels(int nodeCount, int* optPlacements, int n, vector<string> geneNames){
	vector<string> v;
	int count = 0;
	for(int i = 0; i < nodeCount; i++){
		v.push_back("");
	}

	for(int mut=0; mut<n; mut++){
		string toAppend;
		if(v.at(optPlacements[mut]) == ""){
			toAppend = geneNames.at(mut);
			count++;
		}
		else{
			toAppend = ", " + geneNames.at(mut);
			count++;
		}
		//cout << "        " << j << "\n";
		//cout << "                     "<< optPlacements[j] << "\n";
		v.at(optPlacements[mut]) += toAppend;
	}
	if(v.at(nodeCount-1) == ""){
		v.at(nodeCount-1) = "root";
	}
	for(int i = 0; i < nodeCount; i++){
		if(v.at(i).find(" ") != string::npos){
			v.at(i) = "\"" + v.at(i) + "\"";
		}
	}
	//cout << "added mutations " << count << "\n";
	return v;
}

/* returns the lca of a node that has a non-empty label, the root is assumed to always have a label */
int getLcaWithLabel(int node, int* parent, vector<string> label, int nodeCount){
	int root = nodeCount -1;
	int p = parent[node];;
	while(p != root && label[p]==""){
		p = parent[p];
	}
	return p;
}

std::string getGraphVizBinTree(int* parents, int nodeCount, int m, vector<string> label){
	std::stringstream content;
	content << "digraph G {\n";
	content << "node [color=deeppink4, style=filled, fontcolor=white, fontsize=20, fontname=Verdana];\n";
	for(int i=m; i<nodeCount-1; i++){
		if(label[i] != ""){
			int labelledLCA = getLcaWithLabel(i, parents, label, nodeCount);
			content << label[labelledLCA] << " -> " << label[i] << ";\n";
//		if(label[parents[i]] == ""){
//			content  << parents[i] << " -> ";
//		}
//		else{
//			content << label[parents[i]] << " -> ";
//		}
//		if(label[i] == ""){
//		  content  << i << ";\n";
//		}
//		else{
//			content << label[i] << ";\n";

		}
	}
	content << "node [color=lightgrey, style=filled, fontcolor=black];\n";
	for(int i=0; i<m; i++){
		int labelledLCA = getLcaWithLabel(i, parents, label, nodeCount);
		content << label[labelledLCA] << " -> " << "s" << i << ";\n";



//		if(label[parents[i]] == ""){
//			content << parents[i] << " -> ";
//		}
//		else{
//			content << label[parents[i]] << " -> ";
//		}
//
//		content << "s" << i << ";\n";


	}
	content <<  "}\n";
	return content.str();
}



string getMutTreeGraphViz(vector<string> label, int nodeCount, int m, int* parent){
	stringstream nodes;
	stringstream leafedges;
	stringstream edges;
	for(int i=0; i<m; i++){
		if(label.at(i) != ""){
			nodes << "s" << i << "[label=\"s" << i << "\"];\n";                 // si [label="si"];
			nodes        << i << "[label=\"" << label.at(i) << "\"];\n";                 //   i [label="i"];
			leafedges << "s" << i << " -> " << i << ";\n";
			edges <<        i << " -> " << getLcaWithLabel(i, parent, label, nodeCount) << ";\n";
		}
		else{
			nodes << i << "[label=\"s" << i << "\"];\n";
			leafedges << i << " -> " << getLcaWithLabel(i, parent, label, nodeCount) << ";\n";
		}
	}

	stringstream str;

	str << "digraph g{\n";
	str << nodes.str();
	str << "node [color=deeppink4, style=filled, fontcolor=white];	\n";
	str << edges.str();
	str << "node [color=lightgrey, style=filled, fontcolor=black];  \n";
	str << leafedges.str();
	str << "}\n";
	return str.str();
}

/* writes the given string to file */
void writeToFile(string content, string fileName){
	ofstream outfile;
	outfile.open (fileName.c_str());
	outfile << content;
	outfile.close();
}

/* creates the content for the GraphViz file from a parent vector, using numbers as node labels (from 1 to n+1) */
std::string getGraphVizFileContentNumbers(int* parents, int n){
	std::stringstream content;
	content << "digraph G {\n";
	content << "node [color=deeppink4, style=filled, fontcolor=white];\n";
	for(int i=0; i<n; i++){
		content << parents[i]+1  << " -> "  << i+1 << ";\n";      // plus 1 to start gene labeling at 1 (instead of 0)
	}
	content <<  "}\n";
	return content.str();
}


/* creates the content for the GraphViz file from a parent vector, using the gene names as node labels */
std::string getGraphVizFileContentNames(int* parents, int n, vector<string> geneNames, bool attachSamples, bool** ancMatrix, int m, double** logScores, int** dataMatrix){
	std::stringstream content;
	content << "digraph G {\n";
	content << "node [color=deeppink4, style=filled, fontcolor=white];\n";

	for(int i=0; i<n; i++){
		content << geneNames[parents[i]] << " -> "  << geneNames[i]  << ";\n";
	}

	if(attachSamples==true){

		content << "node [color=lightgrey, style=filled, fontcolor=black];\n";
		std::string attachment = getBestAttachmentString(ancMatrix, n, m, logScores, dataMatrix, geneNames);
		content << attachment;
	}
	content <<  "}\n";
	return content.str();
}

/* creates the attachment string for the samples, the optimal attachment points are recomputed from scratch based on error log Scores */
std::string getBestAttachmentString(bool ** ancMatrix, int n, int m, double** logScores, int** dataMatrix, vector<string> geneNames){
	bool** matrix = attachmentPoints(ancMatrix, n, m, logScores, dataMatrix);
	std::stringstream a;
	for(int i=0; i<=n; i++){
		for(int j=0; j<m; j++){
			if(matrix[i][j]==true){
				a << geneNames[i] << " -> s" << j << ";\n";
			}
		}
	}
	return a.str();
}

/* This is a re-computation of the best attachment points of the samples to a tree for printing the tree with attachment points */
/*   gets an ancestor matrix and returns a bit matrix indicating the best attachment points of each sample based on the error log scores */
bool** attachmentPoints(bool ** ancMatrix, int n, int m, double** logScores, int** dataMatrix){

    double treeScore = 0.0;
    bool ** attachment = init_boolMatrix(n+1, m, false);
  	for(int sample=0; sample<m; sample++){       // foreach sample
  		double bestAttachmentScore = 0.0;     // currently best score for attaching sample
  		for(int gene=0; gene<n; gene++){   // start with attaching node to root (no genes mutated)
  			bestAttachmentScore += logScores[dataMatrix[sample][gene]][0];
  		}
  		for(int parent=0; parent<n; parent++){      // try all attachment points (genes)
  		    double attachmentScore=0.0;
  		    for(int gene=0; gene<n; gene++){     // sum up scores for each gene, score for zero if gene is not an ancestor of parent, score for one else wise
  		    	attachmentScore += logScores[dataMatrix[sample][gene]][ancMatrix[gene][parent]];
  		    }
  		    if(attachmentScore > bestAttachmentScore){
  		        bestAttachmentScore = attachmentScore;
  		    }
  		}
  		for(int parent=0; parent<n; parent++){      // try all attachment points (genes)
  		 	double attachmentScore=0.0;
  		 	for(int gene=0; gene<n; gene++){     // sum up scores for each gene, score for zero if gene is not an ancestor of parent, score for one else wise
  		 		attachmentScore += logScores[dataMatrix[sample][gene]][ancMatrix[gene][parent]];
  		 	}
  		  	if(attachmentScore == bestAttachmentScore){
  		  		attachment[parent][sample] = true;
  		  	}
  		}
  		bool rootAttachment = true;
  		for(int parent=0; parent<n; parent++){
  			if(attachment[parent][sample] == true){
  				rootAttachment = false;
  				break;
  			}
  		}
  		if(rootAttachment == true){
  			attachment[n][sample] = true;
  		}
  		treeScore += bestAttachmentScore;
  	}
  	return attachment;
}


/* prints all trees in list of optimal trees to the console, first as parent vector, then as GraphViz file */
void printParentVectors(vector<bool**> optimalTrees, int n, int m, double** logScores, int** dataMatrix){
	for(int i=0; i<optimalTrees.size(); i++){
		int* parents = ancMatrixToParVector(optimalTrees[i], n);
		print_intArray(parents,n);
		//print_boolMatrix(attachmentPoints(optimalTrees[i], n, m, logScores, dataMatrix), n, m);
		printGraphVizFile(parents, n);
	}
}


/* prints the GraphViz file for a tree to the console */
void printGraphVizFile(int* parents, int n){
	cout << "digraph G {\n";
	cout << "node [color=deeppink4, style=filled, fontcolor=white];\n";
	for(int i=0; i<n; i++){
		cout << parents[i] << " -> " << i << "\n";
	}
	cout << "}\n";
}

void printSampleTrees(vector<int*> list, int n, string fileName){
	if(list.size()==0){ return;}
	std::stringstream a;
	for(int i=0; i<list.size(); i++){
		for(int j=0; j<n; j++){
			a << list[i][j];
			if(j<n-1){
				a  << " ";
			}
		}
		a << "\n";
	}
	writeToFile(a.str(), fileName);
	cout << "Trees written to: " << fileName;
}

/* prints the score of the tree predicted by the Kim&Simon approach for the given error log scores */
void printScoreKimSimonTree(int n, int m, double** logScores, int** dataMatrix, char scoreType){
	int parent[] = {2, 4, 17, 2, 9, 9, 2, 2, 4, 18, 2, 1, 2, 2, 9, 2, 2, 11};
	double KimSimonScore = scoreTree(n, m, logScores, dataMatrix, scoreType, parent, -DBL_MAX);
	cout.precision(20);
	//cout << "KimSimonScore: " << KimSimonScore << "\n";
}



//// mcmcBinTreeMove.cpp

/* proposes a new binary tree by a single move from the current binary tree based on the move probabilities */
/* the old tree is kept as currTree, the new one is stored as propTreeParVec */
int* proposeNextBinTree(std::vector<double> moveProbs, int m, int* currTreeParVec, bool** currTreeAncMatrix){

	int movetype = sampleRandomMove(moveProbs);      // pick the move type according to move probabilities
	int parVecLength = (2*m)-2;               // 2m-1 nodes, but the root has no parent

	//cout << "move prob 0: " << moveProbs[0] << "\n";
	//cout << "move prob 1: " << moveProbs[1] << "\n";
	//cout << "move prob 2: " << moveProbs[2] << "\n";
	vector<vector<int> >childLists = getChildListFromParentVector(currTreeParVec, parVecLength);
	int* propTreeParVec  = deepCopy_intArray(currTreeParVec, parVecLength);

	if(movetype==1){                                                       /* type 1: prune and re-attach */
		//cout << "move type is prune and re-attach in binary tree\n";
		int v = pickNodeToMove(currTreeParVec, parVecLength);
		int p = currTreeParVec[v];
		int sib = getSibling(v, currTreeParVec, childLists);             // get the sibling of node v and attach it to the
		propTreeParVec[sib] = currTreeParVec[p];                         // grandparent of v, as the parent of v is moved along with v

		std::vector<int> possibleSiblings = getNonDescendants(currTreeAncMatrix, p, parVecLength);    // get list of possible new siblings of v

		if(possibleSiblings.size()==0){
			cerr << "Error: No new sibling found for node " << v << " for move type 1 in binary tree.\n"; // Should never occur. Something wrong with the tree.
			printGraphVizFile(currTreeParVec, parVecLength);
		}

		int newSibling = possibleSiblings[pickRandomNumber(possibleSiblings.size())]; // pick a new sibling from remaining tree (root can not be a sibling)
		propTreeParVec[newSibling] = p;                                               // make the new sibling a child of v's parent
		propTreeParVec[p] = currTreeParVec[newSibling];                            // make the parent of v the child of the new sibling's former parent
	}
    else{                                                                 /* type 2: swap two node labels  */
    	//cout << "move type is swap node labels in binary tree\n";
    	int v =  rand() % m;                                            // get random leaf to swap (only the first m nodes are leafs)
    	int w =  rand() % m;                                            // get second random leaf to swap
    	propTreeParVec[v] = currTreeParVec[w];                         // and just swap parents
    	propTreeParVec[w] = currTreeParVec[v];
    }
    return propTreeParVec;
}


/* returns a node where the prune and re-attach step starts */
int pickNodeToMove(int* currTreeParentVec, int parentVectorLength){
	bool validNode = false;
	int rootId = parentVectorLength;
	int v;
	while(!validNode){
		v = pickRandomNumber(parentVectorLength);   // pick a node for the prune and re-attach step;
		if(currTreeParentVec[v]!=rootId){               // it has to be a node whose parent is not the root, as node and parent are moved together
			return v;
		}                                      // for a binary tree with more than two leafs this can not be an infinite loop
	}
	return v;
}


/* returns the (unique) sibling of node v. The sibling has to exist because tree is binary */
int getSibling(int v, int* currTreeParVec, vector<vector<int> > &childLists){

	if(childLists.at(currTreeParVec[v]).at(0) != v){
		return childLists.at(currTreeParVec[v]).at(0);
	}
	else{
		return childLists.at(currTreeParVec[v]).at(1);
	}
}

//// mcmcTreeMode.cpp

int* proposeNewTree(vector<double> moveProbs, int n, bool** currTreeAncMatrix, int* currTreeParentVec, double& nbhcorrection){

	int* propTreeParVec = NULL;
	int movetype = sampleRandomMove(moveProbs);      // pick the move type
	//cout << "move type: " << movetype << "\n";
	nbhcorrection = 1;                               // reset the neighbourhood correction

	if(movetype==1){       /* prune and re-attach */
		//cout << "move type is prune and reattach\n";
		int nodeToMove = pickRandomNumber(n);   // pick a node to move with its subtree
		std::vector<int> possibleparents = getNonDescendants(currTreeAncMatrix, nodeToMove, n);      // possible attachment points
		int newParent = choseParent(possibleparents, n);                                             // randomly pick a new parent among available nodes, root (n+1) is also possible parent
		propTreeParVec = getNewParentVecFast(currTreeParentVec, nodeToMove, newParent, n);           // create new parent vector
	}
    else if(movetype==2){   /* swap two node labels  */
    	//cout << "move type is swap node labels\n";
    	int* nodestoswap = sampleTwoElementsWithoutReplacement(n);
        propTreeParVec    = getNewParentVec_SwapFast(currTreeParentVec, nodestoswap[0], nodestoswap[1], n);
        delete [] nodestoswap;
    }
    else if(movetype==3){    /*  swap two subtrees  */
	   //cout << "move type is swap subtrees\n";
        int* nodestoswap = sampleTwoElementsWithoutReplacement(n);                    // pick the node that will be swapped
        nodestoswap = reorderToStartWithDescendant(nodestoswap, currTreeAncMatrix);   // make sure we move the descendant first (in case nodes are in same lineage)
        int nodeToMove = nodestoswap[0];                                              // now we move the first node chosen and its descendants
        int nextnodeToMove = nodestoswap[1];                                          // next we need to move the second node chosen and its descendants
        delete [] nodestoswap;

        if(currTreeAncMatrix[nextnodeToMove][nodeToMove]==0){                      // the nodes are in different lineages -- simple case

        	propTreeParVec = deepCopy_intArray(currTreeParentVec, n);         // deep copy of parent matrix to keep old one
        	propTreeParVec[nodeToMove] = currTreeParentVec[nextnodeToMove];        // and exchange the parents of the nodes
        	propTreeParVec[nextnodeToMove] = currTreeParentVec[nodeToMove];
        }
        else{                                                                      // the nodes are in the same lineage -- need to avoid cycles in the tree
        	propTreeParVec = deepCopy_intArray(currTreeParentVec, n);         // deep copy of parent vector to keep old one
        	propTreeParVec[nodeToMove] = currTreeParentVec[nextnodeToMove];        // lower node is attached to the parent of the upper node
        	std::vector<int> descendants     = getDescendants(currTreeAncMatrix, nodeToMove, n);   // all nodes in the subtree of the lower node
        	bool** propTreeAncMatrix = parentVector2ancMatrix(propTreeParVec, n);
        	std::vector<int> nextdescendants = getDescendants(propTreeAncMatrix, nextnodeToMove, n);
        	free_boolMatrix(propTreeAncMatrix);
        	propTreeParVec[nextnodeToMove] = descendants[pickRandomNumber(descendants.size())];  // node to move is attached to a node chosen uniformly from the descendants of the first node
        	nbhcorrection = 1.0*descendants.size()/nextdescendants.size(); // neighbourhood correction needed for MCMC convergence, but not important for simulated annealing
        }
    }
    return propTreeParVec;
}


/* picks a parent randomly from the set of possible parents, this set includes the root (n+1) */
int choseParent(std::vector<int> &possibleParents, int root){
	possibleParents.push_back(root);                           // add root, as it is also possible attachement point
    int chosenParentPos = pickRandomNumber(possibleParents.size());  // choose where to append the subtree
    int newParent = possibleParents[chosenParentPos];
	possibleParents.pop_back();    // remove root from list of possible parents as it is treated as special case later on
	return newParent;
}



/* creates the new parent vector after pruning and reattaching subtree */
int* getNewParentVecFast(int* currTreeParentVec, int nodeToMove, int newParent, int n){
	int* propTreeParVec = deepCopy_intArray(currTreeParentVec, n);        // deep copy of parent matrix to keep old one
    propTreeParVec[nodeToMove] = newParent;                       // only the parent of the moved node changes
    return propTreeParVec;
}


/* creates the new parent vector after swapping two nodes */
int* getNewParentVec_SwapFast(int* currTreeParentVec, int first, int second, int n){

	int* propTreeParVec = deepCopy_intArray(currTreeParentVec, n);
    for(int i=0; i<n; i++){           // set vector of proposed parents
      if(propTreeParVec[i] == first && i!=second){
      	propTreeParVec[i] = second;              // update entries with swapped parents
      }
      else if(propTreeParVec[i] == second && i!=first){
      	propTreeParVec[i] = first;
      }
    }
    int temp = propTreeParVec[first];
    propTreeParVec[first] = propTreeParVec[second];  // update parents of swapped nodes
    propTreeParVec[second] = temp;
    if(propTreeParVec[first]==first){propTreeParVec[first]=second;}    // this is needed to ensure that tree is connected, the above fails
    if(propTreeParVec[second]==second){propTreeParVec[second]=first;}  // if first is parent of second, or vice versa
    return propTreeParVec;
}

/* re-orders the nodes so that the descendant is first in case the nodes are in the same lineage */
int* reorderToStartWithDescendant(int* nodestoswap, bool** currTreeAncMatrix){
    if(currTreeAncMatrix[nodestoswap[0]][nodestoswap[1]]==true){  // make sure we move the descendent first
        int temp = nodestoswap[0];
        nodestoswap[0] = nodestoswap[1];
        nodestoswap[1] = temp;
    }
    return nodestoswap;
}


/* creates the new parent vector after swapping two nodes */
int* getNewParentVec_Swap(int* currTreeParentVec, int first, int second, int n, int* propTreeParVec){
    for(int i=0; i<n; i++){           // set vector of proposed parents
      if(propTreeParVec[i] == first && i!=second){
      	propTreeParVec[i] = second;              // update entries with swapped parents
      }
      else if(propTreeParVec[i] == second && i!=first){
      	propTreeParVec[i] = first;
      }
    }

    int temp = propTreeParVec[first];
    propTreeParVec[first] = propTreeParVec[second];  // update parents of swapped nodes
    propTreeParVec[second] = temp;
    if(propTreeParVec[first]==first){propTreeParVec[first]=second;}    // this is needed to ensure that tree is connected, the above fails
    if(propTreeParVec[second]==second){propTreeParVec[second]=first;}  // if first is parent of second, or vice versa
    return propTreeParVec;
}


/* creates the new ancestor matrix after swapping two nodes, old matrix is kept */
bool** getNewAncMatrix_Swap(bool** currTreeAncMatrix, int first, int second, int n, bool** propTreeAncMatrix){

    for(int i=0; i<n; i++){                                       // swap columns
    	bool temp = propTreeAncMatrix[i][first];
      	propTreeAncMatrix[i][first] = propTreeAncMatrix[i][second];
      	propTreeAncMatrix[i][second] = temp;
      }
      for(int i=0; i<n; i++){                                // swap rows
      	bool temp = propTreeAncMatrix[first][i];
      	propTreeAncMatrix[first][i] = propTreeAncMatrix[second][i];
      	propTreeAncMatrix[second][i] = temp;
      }
     return propTreeAncMatrix;
 }


/* creates the new parent vector after pruning and reattaching subtree */
int* getNewParentVec(int* currTreeParentVec, int nodeToMove, int newParent, int n, int *propTreeParVec){
    propTreeParVec[nodeToMove] = newParent;                       // only the parent of the moved node changes
    return propTreeParVec;
}


/* creates the new ancestor matrix after pruning and reattaching subtree, old matrix is kept */
bool** getNewAncMatrix(bool** currTreeAncMatrix, int newParent, std::vector<int> descendants, std::vector<int> possibleParents, int n, bool** propTreeAncMatrix){

    if(newParent<n){    // replace the non-descendants of the node and its descendants by the ancestors of the new parent
		for(int i=0; i<possibleParents.size(); i++){
  			for(int j=0; j<descendants.size(); j++){
  				propTreeAncMatrix[possibleParents[i]][descendants[j]] = currTreeAncMatrix[possibleParents[i]][newParent];
  			}
  		}
    }
    else
    {     // if we attach to the root, then they have no further ancestors
        for(int i=0; i<possibleParents.size(); i++){
  			for(int j=0; j<descendants.size(); j++){
        		propTreeAncMatrix[possibleParents[i]][descendants[j]] = 0;
        	}
        }
    }
    return propTreeAncMatrix;
}





//// mcmc.cpp
unsigned int optCount = 0;    // number of steps spent in optimal tree after burn-in phase
double burnInPhase = 0.25;    // first quarter of steps are burn in phase



/* This runs the MCMC for learning the tree and beta, or only the tree with a fixed beta, it samples from the posterior and/or records the optimal trees/beta */
std::string runMCMCbeta(vector<struct treeBeta>& bestTrees, double* errorRates, int noOfReps, int noOfLoops, double gamma1, vector<double> moveProbs, int n, int m, int** dataMatrix, char scoreType, int* trueParentVec, int step, bool sample, double chi, double priorSd, bool useTreeList, char treeType){


	unsigned int optStatesAfterBurnIn = 0;
	int burnIn = noOfLoops*burnInPhase;
	int parentVectorSize = n;
	if(treeType=='t'){parentVectorSize = (2*m)-2;}                     // transposed case: binary tree, m leafs and m-1 inner nodes, root has no parent
	double betaPriorMean = errorRates[1] + errorRates[2];             // AD1 + AD2 the prior mean for AD error rate
	double betaPriorSd   = priorSd;                                     //  prior sd for AD error rate
	double bpriora = ((1-betaPriorMean)*betaPriorMean*betaPriorMean/(betaPriorSd*betaPriorSd)) - betaPriorMean;     // <-10.13585344 turn the mean and sd into parameters of the beta distribution
	double bpriorb = bpriora*((1/betaPriorMean)-1);           //<-13.38666556
	double jumpSd = betaPriorSd/chi;                          // chi: scaling of the known error rate for the MH jump; resulting jump sd
	//cout << "betaPriorMean: " << betaPriorMean << "\n";
	//cout << "betaPriorSd:   " << betaPriorSd << "\n";
	//cout << "bpriora:       " << bpriora << "\n";
	//cout << "bpriorb:       " << bpriorb << "\n";
	//printLogScores(logScores);

	int minDistToTrueTree = INT_MAX;             // smallest distance between an optimal tree and the true (if given)
	double bestTreeLogScore = -DBL_MAX;          // log score of T in best (T,beta)
	double bestScore = -DBL_MAX;                 // log score of best combination (T, beta)
	double bestBeta = betaPriorMean;
	stringstream sampleOutput;

	for(int r=0; r<noOfReps; r++){   // repeat the MCMC, start over with random tree each time, only best score and list of best trees is kept between repetitions

		//cout << "MCMC repetition " << r << "\n";
		int*   currTreeParentVec;
		if(treeType=='m'){currTreeParentVec = getRandParentVec(parentVectorSize);}                                     // start MCMC with random tree
		else{             currTreeParentVec = getRandomBinaryTree(m);}                                                 // transposed case: random binary tree

		bool** currTreeAncMatrix =  parentVector2ancMatrix(currTreeParentVec,parentVectorSize);
		double** currLogScores = getLogScores(errorRates[0], errorRates[1], errorRates[2], errorRates[3]);           // compute logScores of conditional probabilities
		double currBeta = betaPriorMean;                                                                                  // the current AD rate
		double currTreeLogScore;
		if(treeType=='m'){ currTreeLogScore = scoreTreeAccurate( n, m, currLogScores, dataMatrix, scoreType, currTreeParentVec);}
		else{              currTreeLogScore = getBinTreeScore(dataMatrix, n, m, currLogScores, currTreeParentVec);}
		double currBetaLogScore = (moveProbs[0]==0) ? 0.0 : logBetaPDF(currBeta, bpriora, bpriorb);                     // zero if beta is fixed
		double currScore = currTreeLogScore+currBetaLogScore;                                                         // combined score of current tree and current beta

		for(int it=0; it<noOfLoops; it++){                                     // run the iterations of the MCMC
        	// if(it % 100000 == 0){ cout << "At mcmc repetition " << r+1 << "/" << noOfReps << ", step " << it << ": best tree score " << bestTreeLogScore << " and best beta " << bestBeta << " and best overall score " << bestScore << "\n";}

        	bool moveAccepted = false;                                           // Is the MCMC move accepted?
        	bool moveChangesBeta = changeBeta(moveProbs[0]);                     // true if this move changes beta, not the tree

        	if(moveChangesBeta){                                                                // new beta is proposed, log scores change tree is copy of current tree
        		double propBeta = proposeNewBeta(currBeta, jumpSd);
        		double** propLogScores = deepCopy_doubleMatrix(currLogScores, 4, 2);
        		updateLogScores(propLogScores, propBeta);
        		double propBetaLogScore = logBetaPDF(propBeta, bpriora, bpriorb);
        		double propTreeLogScore;
        		if(treeType=='m'){ propTreeLogScore = scoreTree( n, m, propLogScores, dataMatrix, scoreType, currTreeParentVec, bestTreeLogScore);}   // compute the new tree score for new beta
        		else{              propTreeLogScore = getBinTreeScore(dataMatrix, n, m, propLogScores, currTreeParentVec);}

        		if (sample_0_1() < exp((propTreeLogScore+propBetaLogScore-currTreeLogScore-currBetaLogScore)*gamma1)){               // the proposed move is accepted
        			moveAccepted = true;
        			free_doubleMatrix(currLogScores);
        		    currTreeLogScore  = propTreeLogScore;                                       // update score of current tree
        		    currBeta = propBeta;                                                        // the current AD rate
        		    currBetaLogScore = propBetaLogScore;
        		    currScore = currTreeLogScore+currBetaLogScore;                          // combined score of current tree and current beta
        		    currLogScores = propLogScores;
        		}
        		else{
        			delete [] propLogScores[0];
        			delete [] propLogScores;
        		}
        	}
        	else{                                   // move changed tree
        		double nbhcorrection = 1.0;
        		int* propTreeParVec;
        		double propTreeLogScore;
        		if(treeType=='m'){ propTreeParVec = proposeNewTree(moveProbs, n, currTreeAncMatrix, currTreeParentVec, nbhcorrection);              // propose new tree and
        		                   propTreeLogScore = scoreTree( n, m, currLogScores, dataMatrix, scoreType, propTreeParVec, bestTreeLogScore);}    //  get the new tree score
        		else{              propTreeParVec = proposeNextBinTree(moveProbs, m, currTreeParentVec, currTreeAncMatrix);
        		                   propTreeLogScore = getBinTreeScore(dataMatrix, n, m, currLogScores, propTreeParVec);}

        		if (sample_0_1() < nbhcorrection*exp((propTreeLogScore-currTreeLogScore)*gamma1)){                    // the proposed tree is accepted
        			moveAccepted = true;
        			free_boolMatrix(currTreeAncMatrix);                                            // discard outdated tree
        			delete[] currTreeParentVec;
        			currTreeAncMatrix = parentVector2ancMatrix(propTreeParVec,parentVectorSize); // update matrix of current tree
        			currTreeParentVec = propTreeParVec;                                         // update parent vector of current tree
        			currTreeLogScore  = propTreeLogScore;                                       // update score of current tree
        			currScore = currTreeLogScore+currBetaLogScore;
        		}
        		else{
        			delete [] propTreeParVec;            // discard proposed tree
        		}
        	}

        	/* If the true tree is given update the smallest distance between a currently best tree and the true tree */
        	if(trueParentVec){
        		minDistToTrueTree = updateMinDistToTrueTree(trueParentVec, currTreeParentVec, parentVectorSize, minDistToTrueTree, currScore, bestScore);
        	}

        	/* If the list of optimal trees is used, update it */
        	if(useTreeList){
        		updateTreeList(bestTrees, currTreeParentVec, parentVectorSize, currScore, bestScore, currBeta);
        	}

        	/* Sample from the posterior if required and past the burn-in phase */
        	if(sample && it>=burnIn && it % step == 0){
        		sampleOutput << sampleFromPosterior(currTreeLogScore, parentVectorSize, currTreeParentVec, moveProbs[0], currBeta, currScore);
        	}

        	/* Update best tree in case we have found a new best one */
        	if(currScore > bestScore){
        		optStatesAfterBurnIn = 0;                    // new opt state found, discard old count
        		bestTreeLogScore = currTreeLogScore;
        		bestScore = currScore;                 // log score of best combination (T, beta)
        		bestBeta = currBeta;
        	}

        	/* Update the number of MCMC steps we spent in an optimal state */
        	if(currScore == bestScore && it>=burnIn){
        		optStatesAfterBurnIn++;
        	}
        }
        delete [] currTreeParentVec;
        free_doubleMatrix(currLogScores);
        free_boolMatrix(currTreeAncMatrix);
	}                                              // last repetition of MCMC done

	unsigned int noStepsAfterBurnin = noOfReps*(noOfLoops-burnIn);
	cout.precision(17);
	//cout << "best log score for tree:\t" << bestTreeLogScore <<  "\n";
	//cout << "#optimal steps after burn-in:\t" << optStatesAfterBurnIn << "\n";
	//cout << "total #steps after burn-in:\t" << noStepsAfterBurnin << "\n";
	//cout << "%optimal steps after burn-in:\t" << (1.0*optStatesAfterBurnIn)/noStepsAfterBurnin << "\n";
	if(moveProbs[0]!=0.0){
		//cout << "best value for beta:\t" << bestBeta << "\n";
		//cout << "best log score for (T, beta):\t" << bestScore << "\n";
	}

	return sampleOutput.str();
}


double logBetaPDF(double x, double bpriora, double bpriorb){
	double logScore = log(tgamma(bpriora+bpriorb))+(bpriora-1)*log(x)+(bpriorb-1)*log(1-x)-log(tgamma(bpriora))-log(tgamma(bpriorb));    // f(x,a,b) = gamma(a+b)/(gamma(a)gamma(b)) * x^(a-1) * (1-x)^(b-1)
	return logScore;
}

/* a new value for the error probability beta is sampled from a normal distribution around the current beta */
double proposeNewBeta(double currBeta, double jumpSd){
	double sampledValue = sampleNormal(0, jumpSd);
	double propBeta = currBeta+sampledValue ;                   //rnorm(1,0,jumpsd)
	if(propBeta < 0){
		propBeta = abs(propBeta);
	}
	if(propBeta > 1){
		propBeta = propBeta - 2*(propBeta-1);
	}
    return propBeta;
}


/* samples a new value for beta from a normal distribution around the current value */
double sampleNormal(double mean, double sd) {
    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1){
    	return sampleNormal(mean, sd);
    }
    double c = sqrt(-2 * log(r) / r);
    double value =  u * c;                       // value times sd and add the mean
    return (value * sd + mean);
}


/* prints out the current tree and beta to sample from the posterior distribution */
string sampleFromPosterior(double currTreeLogScore, int n, int* currTreeParentVec, double betaProb, double currBeta, double currScore){

	std::stringstream content;
	content << currTreeLogScore  << "\t";                 // logscore of current tree
	content << countBranches(currTreeParentVec, n);       // number of branches in current tree
	if(betaProb>0.0){
		content << "\t" << currBeta;                      // current beta
		content << "\t" << currScore;                     // current combined logscore for tree and beta
	}
	content << "\t";
	for(int i=0; i<n; i++){
		content << currTreeParentVec[i] << " ";
	}
	content << "\n";
	return content.str();
}


/* updates the minimum distance between any of the optimal trees and the true tree (if available) */
int updateMinDistToTrueTree(int* trueParentVec, int* currTreeParentVec, int length, int minDistToTrueTree, int currScore, int bestScore){

	int currDistToTrueTree = getSimpleDistance(trueParentVec, currTreeParentVec, length);

	if(currScore >= bestScore){
		return currDistToTrueTree;
	}

	if(currScore == bestScore && currDistToTrueTree < minDistToTrueTree){         // the current tree is closest to the true tree among the current optimal trees
		return currDistToTrueTree;
	}

	return minDistToTrueTree;
}



/* Returns the distance between two trees, where the distance is the number of nodes having different parents in the two trees  */
int getSimpleDistance(int* trueVector, int* predVector, int length){
	int dist = 0;
	for(int i=0; i<length; i++){
		if(trueVector[i]!=predVector[i]){
			dist++;
		}
	}
	return dist;
}

/*****    functions for sampling random numbers inside C++  *****/
void initRand(){
	time_t t;
	time(&t);
	srand((unsigned int)t);              // initialize random number generator
	//srand(1);
}


/* This function gets a number of nodes n, and creates a random pruefer code for a rooted tree with n+1 nodes (root is always node n+1) */
int* getRandTreeCode(int n){                // as usual n is the number of mutations

	int nodes = n+1;                        // #nodes = n mutations plus root (wildtype)
	int codeLength = nodes-2;
	int* code = new int[codeLength];
	for(int i=0; i<codeLength; i++){
		code[i] = rand() % nodes;
	}
	return code;
}

bool changeBeta(double prob){
	 double percent = (rand() % 100)+1;    // between 1 and 100
	 if(percent <= prob*100){
		 return true;
	 }
	 return false;
}

int sampleRandomMove(std::vector<double> prob){ // picks randomly one of the tree moves based on the move probabilities

    double percent = (rand() % 100)+1;    // between 1 and 100
    double probSum = prob[1];
    for(int i=1; i<prob.size()-1; i++){    // start at index 1; the probability at prob[0] is for changing the error rate (which is treated separately)
        if(percent <= probSum*100){
          return i;
        }
        probSum += prob[i+1];
    }
    return prob.size()-1;
}


bool samplingByProb(double prob){
	double percent = rand() % 100;
	if(percent <= prob*100){
		return true;
	}
	return false;
}


int* sampleTwoElementsWithoutReplacement(int n){

    int* result = new int[2];
	  result[0] = rand() % n;
	  result[1] = result[0];
    while(result[0]==result[1]){
      result[1] = rand() % n;
    }
	  return result;
}

int pickRandomNumber(int n){

    return (rand() % n);
}

double sample_0_1(){

  //return (((double) rand()+0.5) / ((RAND_MAX+1)));
  return ((double) rand() / RAND_MAX);
}

int getElemFromQueue(int index, std::vector<int> queue){
	int elem = queue.at(index);
	if (index != queue.size() - 1)
	{
		queue[index] = std::move(queue.back());
	}

	//cout << queue.size() << " elements in queue in subroutine\n";
	return elem;
}

// This creates the parent vector of a random binary tree. Entries 0...m-1 are for the leafs.
// Entries m....2m-3 are for the inner nodes except the root, the root has index 2m-2 which has no parent
// and therefore has no entry in the parent vector
int* getRandomBinaryTree(int m){
	int parentCount = (2*m)-2;     // the m leafs have a parent and so have m-2 of the m-1 inner nodes
	int* leafsAndInnerNodesParents = init_intArray(parentCount, -1);

	std::vector<int> queue;
	for(int i=0; i<m; i++){queue.push_back(i);}   // add the m leafs to the queue
	//cout << queue.size() << " elements in queue\n";
	int innerNodeCount = m;
	while(queue.size()>1){
		int pos = pickRandomNumber(queue.size());
		int child1 = queue.at(pos);
		if (pos != queue.size() - 1){queue[pos] = std::move(queue.back());}
		queue.pop_back();
		//cout << queue.size() << " elements in queue\n";

		pos = pickRandomNumber(queue.size());
		int child2 = queue.at(pos);
		if (pos != queue.size() - 1){queue[pos] = std::move(queue.back());}
		queue.pop_back();
		//cout << queue.size() << " elements in queue\n";

		leafsAndInnerNodesParents[child1] = innerNodeCount;
		leafsAndInnerNodesParents[child2] = innerNodeCount;
		queue.push_back(innerNodeCount);
		innerNodeCount++;
	}
	return leafsAndInnerNodesParents;
}


double epsilon = 0.000000000001;  // how much worse than the current best a score can be to still use the accurate score computation


/****     Tree scoring     ****/


/* Computes the score of a new candidate tree. First a fast approximate score is computed, then if the new score is better,   */
/* or slightly worse than the best score so far, a more accurate but more costly score computation is done.                    */
double scoreTree(int n, int m, double** logScores, int** dataMatrix, char type, int* parentVector, double bestTreeLogScore){

	double approx = scoreTreeFast(n, m, logScores, dataMatrix, type, parentVector);   // approximate score

	if(approx > bestTreeLogScore-epsilon){                                                  // approximate score is close to or better
		return scoreTreeAccurate(n, m, logScores, dataMatrix, type, parentVector);   // than the current best score, use accurate
	}                                                                                // score computation

	return approx;                                                              // otherwise the approximate score is sufficient
}



/****     Fast (approximate) tree scoring     ****/

/* computes an approximate score for a tree. This is fast, but rounding errors can occur  */
double scoreTreeFast(int n, int m, double** logScores, int** dataMatrix, char type, int* parentVector){

	double result = -DBL_MAX;
	int* bft = getBreadthFirstTraversal(parentVector, n);   // get breadth first traversal for simple scoring
	                                                        // by updating the parent score
	if(type=='m'){
		result = maxScoreTreeFast(n, m, logScores, dataMatrix, parentVector, bft);  // score by best attachment point per sample
	}
	else if(type=='s'){
		result = sumScoreTreeFast(n, m, logScores, dataMatrix, parentVector, bft);  // score by summing over all attachment points
	}

	delete [] bft;
	return result;
}

/* computes an approximate scoring for a tree using the max attachment score per sample */
double maxScoreTreeFast(int n, int m, double** logScores, int** dataMatrix, int* parent, int* bft){

    double treeScore = 0.0;

  	for(int sample=0; sample<m; sample++){                                                      // for all samples get
  		double* scores = getAttachmentScoresFast(parent,n, logScores, dataMatrix[sample], bft);  // all attachment scores
  		treeScore +=  getMaxEntry(scores, n+1);
  		delete [] scores;
  	}

  	return treeScore;    // sum over the best attachment scores of all samples is tree score
}


/* computes an approximate scoring for a tree summing the score over all attachment points per sample */
double sumScoreTreeFast(int n, int m, double** logScores, int** dataMatrix, int* parent, int* bft){

	double sumTreeScore = 0.0;

	for(int sample=0; sample<m; sample++){
		double* scores = getAttachmentScoresFast(parent, n, logScores, dataMatrix[sample], bft); // attachments scores of sample to each node
		double bestMaxTreeScore = getMaxEntry(scores, n+1);                                     // best of the scores (used to compute with score differences rather than scores)

		double sumScore = 0.0;
		for(int i=0; i<=n; i++){                                                 // sum over all attachment scores, exp is necessary as scores are actually log scores
			sumScore += exp(scores[bft[i]]-bestMaxTreeScore);                   // subtraction of best score to calculate with score differences (smaller values)
		}
		delete [] scores;
		sumTreeScore += log(sumScore)+bestMaxTreeScore;                     // transform back to log scores and change from score differences to actual scores
	}
	return sumTreeScore;
}


/* computes the attachment scores of a sample to all nodes in the tree (except root) */
double* getAttachmentScoresFast(int*parent, int n, double** logScores, int* dataVector, int*bft){

	double* attachmentScore = init_doubleArray(n+1, -DBL_MAX);
	attachmentScore[n] = rootAttachementScore(n, logScores, dataVector);
	for(int i=1; i<=n; i++){                                                              // try all attachment points (nodes in the mutation tree)
		int node = bft[i];
		attachmentScore[node] = attachmentScore[parent[node]];
		attachmentScore[node] -= logScores[dataVector[node]][0];
		attachmentScore[node] += logScores[dataVector[node]][1];
	}
	return attachmentScore;
}

/* computes the log score for attaching a sample to the root node (this score is equal for all trees) */
double rootAttachementScore(int n, double** logScores, int* dataVector){
	double score = 0.0;
	for(int gene=0; gene<n; gene++){                          // sum over log scores for all other nodes in tree
		score += logScores[dataVector[gene]][0];      // none of them is ancestor of the sample as it is attached to root
	}
	return score;
}





/****  accurate score computation (minimizes rounding errors)  ****/

double scoreTreeAccurate(int n, int m, double** logScores, int** dataMatrix, char type, int* parentVector){

	double result = -DBL_MAX;
	int* bft = getBreadthFirstTraversal(parentVector, n);
	if(type=='m'){
		result = maxScoreTreeAccurate(n, m, logScores, dataMatrix, parentVector, bft);
	}
	else if(type=='s'){
		result = sumScoreTreeAccurate(n, m, logScores, dataMatrix, parentVector, bft);
	}

	delete [] bft;
	return result;
}

double maxScoreTreeAccurate(int n, int m, double** logScores, int** dataMatrix, int* parent, int* bft){

    int** treeScoreMatrix = init_intMatrix(4, 2, 0);
  	for(int sample=0; sample<m; sample++){
  		int** bestAttachmentMatrix =  getBestAttachmentScoreAccurate(init_intMatrix(4, 2, 0), parent, n, logScores, dataMatrix[sample], bft);
  		treeScoreMatrix = sumMatrices(treeScoreMatrix, bestAttachmentMatrix, 4, 2);
  		free_intMatrix(bestAttachmentMatrix);
  	}
  	double treeScore = getTrueScore(treeScoreMatrix, logScores);
  	free_intMatrix(treeScoreMatrix);
  	return treeScore;
}

/* computes the log score for the complete tree using the sumScore scheme, where likelihoods of all attachment points of a sample are added */
double sumScoreTreeAccurate(int n, int m, double** logScores, int** dataMatrix, int* parent, int* bft){

	double sumTreeScore = 0.0;

	for(int sample=0; sample<m; sample++){
		sumTreeScore += getSumAttachmentScoreAccurate(parent, n, logScores, dataMatrix[sample], bft);
	}
	return sumTreeScore;
}

/* computes the best attachment score for a sample to a tree */
int** getBestAttachmentScoreAccurate(int** scoreMatrix, int* parent, int n, double** logScores, int* dataVector, int* bft){

	int*** attachmentScoreMatrix = getAttachmentMatrices(parent, n, dataVector, bft);   // matrix to keep attachment scores for each sample (not summing up
	                                                                                            // components to avoid rounding errors)
	double bestScore =  -DBL_MAX;
	int** bestScoreMatrix = NULL;

	for(int i=0; i<n+1; i++){                                                                   // now get true attachment scores and find best score among all attachment points
		double newScore = getTrueScore(attachmentScoreMatrix[i], logScores);
		if(bestScore <= newScore){
			bestScoreMatrix = attachmentScoreMatrix[i];
			bestScore = newScore;
		}
	}
	scoreMatrix = sumMatrices(scoreMatrix, bestScoreMatrix, 4, 2);
	delete_3D_intMatrix(attachmentScoreMatrix, n+1);
	return scoreMatrix;
}

/* computes the sum score for attaching a sample to all nodes */
double getSumAttachmentScoreAccurate(int* parent, int n, double** logScores, int* dataVector, int* bft){

	int*** attachmentScoreMatrix = getAttachmentMatrices(parent, n, dataVector, bft);   // matrix to keep attachment scores for each sample (not summing up
		                                                                                            // components to avoid rounding errors)
	double* attachmentScore = getTrueScores(attachmentScoreMatrix, n, logScores);                   // get the true attachment scores from the attachment matrices
	double bestScore = getMaxEntry(attachmentScore, n+1);                                            // identify best attachment score
	double sumScore = 0.0;
	for(int parent = 0; parent<n+1; parent++){                                                        // get score for attaching to the other nodes in the tree
		sumScore += exp(attachmentScore[parent]-bestScore);
	}
	delete_3D_intMatrix(attachmentScoreMatrix, n+1);
	delete [] attachmentScore;
	return log(sumScore)+bestScore;
}

/* computes the attachment scores of a sample to all nodes in a tree, score is a matrix counting the number of different match/mismatch score types */
int*** getAttachmentMatrices(int* parent, int n, int* dataVector, int* bft){
	int*** attachmentScoreMatrix = new int**[n+1];            // matrix to keep attachment scores for each sample (not summing up components to avoid rounding errors)

	// start with attaching node to root (no genes mutated)
	attachmentScoreMatrix[n] = init_intMatrix(4, 2, 0);
	for(int gene=0; gene<n; gene++){
		attachmentScoreMatrix[n][dataVector[gene]][0]++;
	}

	// now get scores for the other nodes due to bft traversal in an order such that attachment matrix of parent is already filled
	for(int i=1; i<n+1; i++){
		int node = bft[i];
		attachmentScoreMatrix[node] = deepCopy_intMatrix(attachmentScoreMatrix[parent[node]], 4, 2);
		attachmentScoreMatrix[node][dataVector[node]][0]--;
		attachmentScoreMatrix[node][dataVector[node]][1]++;
	}
	return attachmentScoreMatrix;
}

double* getTrueScores(int*** matrix, int n, double** logScores){
	double* scores = new double[n+1];
	for(int node=0; node<=n; node++){
		scores[node] = getTrueScore(matrix[node], logScores);
	}
	return scores;
}


/* computes the attachment score of a sample to a tree from a matrix */
/* representation of the score (counting the # 0->1, 1->0, 0->0, ...) */
double getTrueScore(int** matrix, double** logScores){
	double score = 0.0;
	for(int j=0; j<4; j++){
		for(int k=0; k<2; k++){
			double product = matrix[j][k] * logScores[j][k];
			//cout << "[" << j << "][" << k << "] = " << matrix[j][k] << " * " << logScores[j][k] << "[" << j <<"][" << k << "]\n";
			score = score + product;
		}
	}
	return score;
}

/***********************         Scoring Tables            ***************************/

/* computes a table of the log scores of observing one genotype, given that another genotype */
/* is the true one; for three observed types (+missing observation) and two true states */
double** getLogScores(double FD, double AD1, double AD2, double CC){

  double** logScores = init_doubleMatrix(4, 2, 0.0);
  logScores[0][0] = log(1.0-CC-FD);  // observed 0, true 0
	logScores[1][0] = log(FD);         // observed 1, true 0
	if(CC!=0.0){
		logScores[2][0] = log(CC);         // observed 2, true 0
	}
	else{                                //  to capture case where CC=0, because log(0) = -infinity
		logScores[2][0] = 0.0;           // CC=0 should only occur when there are no 2's (double mutations) in the matrix
	}
	logScores[3][0] = log(1.0);          // value N/A,  true 0
	logScores[0][1] = log(AD1);      // observed 0, true 1
	logScores[1][1] = log(1.0-(AD1+AD2));     // observed 1, true 1
	if(AD2 != 0.0){
		logScores[2][1] = log(AD2);     // observed 2, true 1
	}
	else{
		logScores[2][1] = 0;
	}
	logScores[3][1] = log(1.0);          // value N/A,  true 1
	return logScores;
}

/* updates the log scores after a new AD rate was accepted in the MCMC */
void updateLogScores(double** logScores, double newAD){

	double newAD1 = newAD;      // the default case: there are no homozygous mutation observed
	double newAD2 = 0.0;

	if(logScores[2][1] != 0){          // other case: homozygous mutations were called
		newAD1 = newAD/2;          // for simplicity we set both dropout rates to 1/2 of the new value
		newAD2 = newAD/2;          // learning the rates separately could also be done
	}

	logScores[0][1] = log(newAD1);          // observed 0, true 1
	logScores[1][1] = log(1.0-(newAD));     // observed 1, true 1
	if(newAD2 != 0.0){
		logScores[2][1] = log(newAD2);     // observed 2, true 1
	}
	else{
		logScores[2][1] = 0;
	}
	logScores[3][1] = log(1.0);          // value N/A,  true 1
}



//// scoreBinTree.cpp

/* computes the log likelihood for a single mutation for all subtrees of the binary tree, where the expected */
/* state of the mutation can be either absent or present in the whole subtree (passed as 'state' to the function) */
double* getBinSubtreeScore(bool state, int* bft, vector<vector<int> > &childLists, int mut, int nodeCount, int m, int** obsMutProfiles, double ** logScores){
	double* score = init_doubleArray(nodeCount, 0.0);
	for(int i=nodeCount-1; i>=0; i--){
		int node = bft[i];

		if(node < m){
			score[node] = logScores[obsMutProfiles[node][mut]][state];   // for leafs the score is just P(Dij|Eij)
		}
		else{                                                          // for inner nodes the score is the sum of the scores of the children
			if(childLists.at(node).size()!=2){
				cerr << "Error node " << node << " has " << childLists.at(node).size() << " children\n";  // tree should be binary, but isn't
			}
			score[node] = score[childLists.at(node).at(0)] + score[childLists.at(node).at(1)];
		}
	}
	return score;
}


/* Computes the best log likelihood for placing a single mutation in a given sample tree */
/* Iterates through all nodes as possible placements of the mutation to find the best one */
/* All samples below the placement of the mutation should have it, mutation can also be placed at a leaf, i.e. uniquely to the sample   */
double getBinTreeMutScore(int* bft, vector<vector<int> > &childLists, int mut, int nodeCount, int m, int** obsMutProfiles, double ** logScores){

	double bestScore = -DBL_MAX;
	double* absentScore = getBinSubtreeScore(0, bft, childLists, mut, nodeCount, m, obsMutProfiles, logScores);
	double* presentScore = getBinSubtreeScore(1, bft, childLists, mut, nodeCount, m, obsMutProfiles, logScores);

	for(int p=0; p<nodeCount; p++){
		double score = absentScore[nodeCount-1] - absentScore[p] + presentScore[p];
		bestScore = max(bestScore, score);
	}

	delete [] absentScore;
	delete [] presentScore;
	return bestScore;
}

/* Computes the maximum log likelihood of a binary tree for a given mutation matrix.  */
/* Note: No extra root necessary for binary trees */
double getBinTreeScore(int** obsMutProfiles, int n, int m, double ** logScores, int* parent){

	int nodeCount = (2*m)-1;   // number of nodes in binary tree: m leafs, m-1 inner nodes (includes already the root)
	double sumScore = 0;       // sum of maximal scores of all samples
	vector<vector<int> > childLists = getChildListFromParentVector(parent, nodeCount-1);
	int* bft = getBreadthFirstTraversal(parent, nodeCount-1);

	for(int mut=0; mut<n; mut++){                                                // sum over the optimal scores of each sample
		double score = getBinTreeMutScore(bft, childLists, mut, nodeCount, m, obsMutProfiles, logScores);
		sumScore += score;
	}

	delete [] bft;
	for(int i=0; i<childLists.size(); i++){
		childLists[i].clear();
	}
	childLists.clear();
	//cout << "score: " << sumScore << "\n";
	return sumScore;
}




//// scoreTree.cpp

double** getScores(double FD, double AD1, double AD2, double CC){

  double** scores = init_doubleMatrix(4, 2, 0.0);
  scores[0][0] = 1.0-CC-FD;  // observed 0, true 0
  scores[1][0] = FD;         // observed 1, true 0
  scores[2][0] = CC;         // observed 2, true 0
  scores[3][0] = 1.0;          // value N/A,  true 0
  scores[0][1] = AD1;          // observed 0, true 1
  scores[1][1] = 1.0-(AD1+AD2);     // observed 1, true 1
  scores[2][1] = AD2;               // observed 2, true 1
  scores[3][1] = 1.0;               // value N/A,  true 1
  return scores;
}

void printLogScores(double** logScores){
	cout.precision(70);
	for(int i=0; i<4; i++){
		for(int j=0; j<2; j++){
			cout << logScores[i][j] << "\t";
		}
		cout << "\n";
	}
}

/* Constants and global variables */
double defaultMoveProbs[] = {0.55, 0.4, 0.05};     // moves: change beta / prune&re-attach / swap node labels / swap subtrees
double defaultMoveProbsBin[] = {0.4, 0.6};    // moves: change beta / prune&re-attach / swap leaf labels

double errorRateMove = 0.0;
vector<double> treeMoves;
double chi = 10;
double priorSd = 0.1;
string fileName;      // data file
string outFile;       // the name of the outputfile, only the prefix before the dot
int n;                // number of genes
int m;                // number of samples
char scoreType = 'm';
int rep;            // number of repetitions of the MCMC
int loops;          // number of loops within a MCMC
double gamma1 = 1;
double fd;          // rate of false discoveries (false positives 0->1)
double ad1;          // rate of allelic dropout (false negatives 1->0)
double ad2 = 0.0;         // rate of allelic dropout (2->1)
double cc = 0.0;          // rate of falsely discovered homozygous mutations (0->2)
bool sample = false;
int sampleStep;
bool useGeneNames = false;        // use gene names in tree plotting
string geneNameFile;              // file where the gene names are listed.
bool trueTreeComp = false;      // set to true if true tree is given as parameter for comparison
string trueTreeFileName;        // optional true tree
bool attachSamples = true;       // attach samples to the tree
//bool useFixedSeed = false;      // use a predefined seed for the random number generator
bool useFixedSeed = true;      // use a predefined seed for the random number generator


unsigned int fixedSeed = 1;   // default seed
bool useTreeList = true;
char treeType = 'm';        // the default tree is a mutation tree; other option is 't' for (transposed case), where we have a binary leaf-labeled tree
int maxTreeListSize = -1;   // defines the maximum size of the list of optimal trees, default -1 means no restriction


/*Convert a contiguous 1D array (coming from Python/ctypes) into a 2D matrix*/
int** array2matrix(int *dataArray, int n_cells, int n_genes){

    int** dataMatrix = init_intMatrix(n_cells, n_genes, -1);

    for (int i = 0; i < n_cells; i++) {
        for (int j = 0; j < n_genes; j++) {
            dataMatrix[i][j] = dataArray[i * n_genes + j];
        }
    }
    return dataMatrix;
}


/* Function that exports SCITE interface to Python*/
DLLEXPORT
void scite_export(int *dataArray, int n_cells, int n_genes,
                  int rep, int loops, double fd, double ad1,
                  double ad2, double cc, char treeType, char scoreType,
                  char* bestTrees){

    // Set global variables to correspond to n_cells, n_genes
    m = n_cells;
    n = n_genes;

    // Read data from memory
    int **dataMatrix = array2matrix(dataArray, n_cells, n_genes);
    vector<string> genes = getGeneNames("", n_genes);     // Output genes as IDs;
	std::vector<struct treeBeta> optimalTrees;            // list of optimal tree/beta combinations found by MCMC
	std::string sampleOutput;                             // the samples taken in the MCMC as a string for output

    // These are the one that are not set by default
	int* trueParentVec = NULL;  // True tree is not known in practice

	/**  read parameters and data file  **/
	// readParameters(argc, argv);
	// int** dataMatrix = getDataMatrix(n, m, fileName);
	vector<double> moveProbs = setMoveProbs();
	double* errorRates = getErrorRatesArray(fd, ad1, ad2, cc);

	/* initialize the random number generator, either with a user defined seed, or a random number */
	useFixedSeed? srand(fixedSeed) : initRand();

	/**  Find best scoring trees by MCMC  **/
	sampleOutput = runMCMCbeta(optimalTrees, errorRates, rep, loops, gamma1, moveProbs, n_genes, n_cells,
	    dataMatrix, scoreType, trueParentVec, sampleStep, sample, chi, priorSd, useTreeList, treeType);

    // Fetch first optimal tree only in graph viz format (including sample attachment).
    int parentVectorSize = n;
    int* parentVector = optimalTrees.at(0).tree;
    if(treeType=='t'){parentVectorSize = (2 * m) - 2;}
    double** logScores = getLogScores(fd, ad1, ad2, cc);
    bool** ancMatrix = parentVector2ancMatrix(parentVector, parentVectorSize);

    // GraphViz output
	string output = getGraphVizFileContentNames(parentVector, parentVectorSize, genes,
		    attachSamples, ancMatrix, m, logScores, dataMatrix);
    strcpy(bestTrees, output.c_str());

	// Write trees to output pointer
	delete [] logScores[0];
	delete [] logScores;
	delete [] errorRates;
	free_intMatrix(dataMatrix);
	emptyVectorFast(optimalTrees, n);
}



int* getParentVectorFromGVfile(string fileName, int n){
	int* parentVector = new int[n];
	std::vector<std::string> lines;
	std::ifstream file(fileName.c_str());
	std::string line;
	while ( std::getline(file, line) ) {
	    if ( !line.empty() )
	        lines.push_back(line);
	}
	for(int i=0; i < lines.size(); i++){

		std::size_t found = lines[i].find(" -> ");
		if (found!=std::string::npos){
			int parent = atoi(lines[i].substr(0, found).c_str());
			int child = atoi(lines[i].substr(found+3).c_str());
			parentVector[child-1] = parent-1;
	   }
	}
	return parentVector;
}


int getMinDist(int* trueVector, std::vector<bool**> optimalTrees, int n){
	int minDist = n+1;
	for(int i=0; i<optimalTrees.size(); i++){
		int dist = getSimpleDistance(trueVector, ancMatrixToParVector(optimalTrees.at(i), n), n);
		minDist = min(minDist, dist);
	}
	return minDist;
}


vector<double> setMoveProbs(){
	vector<double> moveProbs;

	moveProbs.push_back(errorRateMove);

	if(treeMoves.size()==0){                                       // use default probabilities
		if(treeType == 'm'){
			moveProbs.push_back(defaultMoveProbs[0]);
			moveProbs.push_back(defaultMoveProbs[1]);
			moveProbs.push_back(defaultMoveProbs[2]);
		}
		else{
			moveProbs.push_back(defaultMoveProbsBin[0]);
			moveProbs.push_back(defaultMoveProbsBin[1]);
		}
	}
	else{                                                                            // use probabilities from command line
		double sum = 0.0;
		for(int i=0; i< treeMoves.size(); i++){ sum += treeMoves[i]; }
		if(sum != 1.0){
			cerr << "move probabilities do not sum to 1.0, recalculating probabilities\n";     // normalize to sum to one
			for(int i=0; i< treeMoves.size(); i++){
				treeMoves[i] = treeMoves[i]/sum;
			}
			//cout << "new move probabilities:";
			//for(int i=0; i< treeMoves.size(); i++){ cout << " " << treeMoves[i];}
			//cout << "\n";
		}
		for(int i=0; i< treeMoves.size(); i++){
			moveProbs.push_back(treeMoves[i]);
		}
	}
	treeMoves.clear();
	return moveProbs;
}

int** getDataMatrix(int n, int m, string fileName){

    int** dataMatrix = init_intMatrix(n, m, -1);

    ifstream in(fileName.c_str());

    if (!in) {
    	cout << "2 Cannot open file " << fileName << "\n";
      cout << fileName;
      cout << "\n";
      return NULL;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            in >> dataMatrix[i][j];
        }
    }

    in.close();
    int** transposedMatrix = transposeMatrix(dataMatrix, n, m);
    free_intMatrix(dataMatrix);

    return transposedMatrix;
}


vector<string> getGeneNames(string fileName, int nOrig){

	vector<string> v;
	ifstream in(fileName.c_str());


	n = nOrig;

	if (!in) {
		//cout << "Cannot open gene names file " << fileName << ", ";
	    //cout << "using ids instead.\n";
	    vector<string> empty;
	    for(int i=0; i<=n; i++){
	    	stringstream id;
	    	id << i+1;
	    	empty.push_back(id.str());
	    }
	    return empty;
	}

	for (int i = 0; i < nOrig; i++) {
		string temp;
	    in >> temp;
	    v.push_back(temp);
	}
	v.push_back("Root"); // the root
	return v;
}


double* getErrorRatesArray(double fd, double ad1, double ad2, double cc){
	double* array = new double[4];
	array[0] = fd;
	array[1] = ad1;
	array[2] = ad2;
	array[3] = cc;
	return array;
}



// Empty python module definition
#include "Python.h"

static PyModuleDef _scite_module = {
	PyModuleDef_HEAD_INIT,
	"_scite",
	NULL,
	-1,
};


PyMODINIT_FUNC
PyInit__scite(void) {
	PyObject * mod;
	mod = PyModule_Create(&_scite_module);
	if (mod == NULL)
		return NULL;
	return mod;
}
