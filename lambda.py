import numpy as np
import ROOT
import sys
import matplotlib.pyplot as plt
from ROOT import *


tree_sig = ROOT.RDataFrame("BTo3Mu",'/home/users/parolia/pre_selection/final_selection/old/weighted/Sig.root')
tree_bkg = ROOT.RDataFrame("BTo3Mu",'/home/users/parolia/pre_selection/final_selection/old/weighted/Bkg.root')
tree_tau = ROOT.RDataFrame("BTo3Mu",'/home/users/parolia/pre_selection/final_selection/old/weighted/Sig_tau.root')
tree_mu = ROOT.RDataFrame("BTo3Mu",'/home/users/parolia/pre_selection/final_selection/old/weighted/Sig_mu.root')
                                                                 
low = [3,2,1,5,0,0.97,0,0.97,5,2,0.5,-0.01]
high = [20,20,20,20,0.1,1.0,0.1,1.00,40,6,1.0,0.01]

#low = [0,0,0,0,0,0.975,0,0.975,0,2,0,-0.03]
#high = [30,30,30,30,1,1.005,1,1.005,80,8.0,2,0.03]

var = ['mu1pt','mu2pt','kpt','jpsi_pt','jpsivtx_svprob','jpsivtx_cos2D','bvtx_svprob','bvtx_cos2D','Bpt_reco','Bmass','DR_jpsimu','ip3d']


sol = sys.argv[1]
gen = sys.argv[2]
#num = sys.argv[3]

lam = ['0p25', '0p50', '0p75']
#lam = ['no_lam', '0p0', '0p25', '0p50', '0p75', '1p0']
#x = np.empty(len(lam))
x =[] 

for n in range(len(lam)):                                                
#data_3 = []
    FileName = "average_" + sol + "_" + gen + "_" + lam[n] +".npy"
    x.append(np.load(FileName))
values = np.asarray(x).T
#print (x)
#print (values)



for i in range(12):

    c = ROOT.TCanvas()
    c.cd()
   
    hs = tree_sig.Histo1D(("hsig", "Distribution of %s_sig with cuts"%var[i], 30, low[i], high[i]), '%s'%var[i])
    hb = tree_bkg.Histo1D(("hbkg", "Distribution of %s_bkg with cuts"%var[i], 30, low[i], high[i]), '%s'%var[i])
    ht = tree_tau.Histo1D(("htau", "Distribution of %s_tau with cuts"%var[i], 30, low[i], high[i]), '%s'%var[i])
    hm = tree_mu.Histo1D(("hmu", "Distribution of %s_muon with cuts"%var[i], 30, low[i], high[i]), '%s'%var[i])

    h = hs.Clone()
    k = hb.Clone()
    t = ht.Clone()
    m = hm.Clone()

    h.SetLineColor(ROOT.kBlack)
    k.SetLineColor(ROOT.kRed)
    t.SetLineColor(ROOT.kGreen)
    m.SetLineColor(ROOT.kBlue)
             
    h.SetTitle("%s Distribution"%var[i])
    h.GetYaxis().SetTitle("Events")
    h.GetXaxis().SetTitle("%s"%var[i])

    if (i > 3 and i < 8):
        c.SetLogy(True)

    maxValHisto = h.GetMaximum()
    h.SetMaximum(1.5 * maxValHisto)
    k.Scale(h.Integral()/k.Integral())
    t.Scale(h.Integral()/t.Integral())
    m.Scale(h.Integral()/m.Integral())
    h.DrawCopy()
    k.DrawCopy("SAME")
    t.DrawCopy("SAME")
    m.DrawCopy("SAME")



    
    if (i == 11):

        cut_1 = float(values[i,0])  
        graph_1 = TGraph(3)                                  
        #set three points of the graph                           
        graph_1.SetPoint(0, cut_1, 0)                                       
        graph_1.SetPoint(1, cut_1, maxValHisto)                             
        graph_1.SetPoint(2, cut_1, 30000)                         
        graph_1.SetLineColor(6)                                            
        graph_1.Draw("same")

        cut_2 = float(values[i,1])  
        graph_2 = TGraph(3)                                  
        #set three points of the graph                           
        graph_2.SetPoint(0, cut_2, 0)                                      
        graph_2.SetPoint(1, cut_2, maxValHisto)                            
        graph_2.SetPoint(2, cut_2, 30000)                         
        graph_2.SetLineColor(7)                                            
        graph_2.Draw("same")
        
        cut_3 = float(values[i,2])  
        graph_3 = TGraph(3)                                  
        #set three points of the graph                           
        graph_3.SetPoint(0, cut_3, 0)                                      
        graph_3.SetPoint(1, cut_3, maxValHisto)                             
        graph_3.SetPoint(2, cut_3, 30000)                         
        graph_3.SetLineColor(8)                                            
        graph_3.Draw("same")
        

        cut2_1 = float(values[i+1,0])  
        graph2_1 = TGraph(3)                                  
        #set three points of the graph                           
        graph2_1.SetPoint(0, cut2_1, 0)                                       
        graph2_1.SetPoint(1, cut2_1, maxValHisto)                             
        graph2_1.SetPoint(2, cut2_1, 30000)                         
        graph2_1.SetLineColor(6)                                            
        graph2_1.Draw("same")

        cut2_2 = float(values[i+1,1])  
        graph2_2 = TGraph(3)                                  
        #set three points of the graph                           
        graph2_2.SetPoint(0, cut2_2, 0)                                      
        graph2_2.SetPoint(1, cut2_2, maxValHisto)                            
        graph2_2.SetPoint(2, cut2_2, 30000)                         
        graph2_2.SetLineColor(7)                                            
        graph2_2.Draw("same")
        
        cut2_3 = float(values[i+1,2])  
        graph2_3 = TGraph(3)                                  
        #set three points of the graph                           
        graph2_3.SetPoint(0, cut2_3, 0)                                      
        graph2_3.SetPoint(1, cut2_3, maxValHisto)                             
        graph2_3.SetPoint(2, cut2_3, 30000)                         
        graph2_3.SetLineColor(8)                                            
        graph2_3.Draw("same")
        
    else:
        
        cut_1 = float(values[i,0])  
        graph_1 = TGraph(3)                                  
        #set three points of the graph                           
        graph_1.SetPoint(0, cut_1, 0)                                       
        graph_1.SetPoint(1, cut_1, maxValHisto)                             
        graph_1.SetPoint(2, cut_1, 30000)                         
        graph_1.SetLineColor(6)                                            
        graph_1.Draw("same")

        cut_2 = float(values[i,1])  
        graph_2 = TGraph(3)                                  
        #set three points of the graph                           
        graph_2.SetPoint(0, cut_2, 0)                                      
        graph_2.SetPoint(1, cut_2, maxValHisto)                            
        graph_2.SetPoint(2, cut_2, 30000)                         
        graph_2.SetLineColor(7)                                            
        graph_2.Draw("same")
        
        cut_3 = float(values[i,2])  
        graph_3 = TGraph(3)                                  
        #set three points of the graph                           
        graph_3.SetPoint(0, cut_3, 0)                                      
        graph_3.SetPoint(1, cut_3, maxValHisto)                             
        graph_3.SetPoint(2, cut_3, 30000)                         
        graph_3.SetLineColor(8)                                            
        graph_3.Draw("same")
        
    #hsig = TH1D* 
    
    legend = TLegend(0.65, 0.65, 0.9, 0.9)
    legend.AddEntry(h, "sig", "l")
    legend.AddEntry(k, "bkg", "l")
    legend.AddEntry(t, "tau", "l")
    legend.AddEntry(m, "muon", "l")
    legend.AddEntry(graph_1, "cut($\lambda$=0.25)", "l")
    legend.AddEntry(graph_2, "cut($\lambda$=0.50)", "l")
    legend.AddEntry(graph_3, "cut($\lambda$=0.75)", "l")
    legend.Draw()
    
    c.SaveAs('%s_cuts.png'%var[i])

