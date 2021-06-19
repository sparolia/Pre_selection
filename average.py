import numpy as np
import ROOT
import sys
import matplotlib.pyplot as plt
from ROOT import *
                           
sol = sys.argv[1]
gen = sys.argv[2]
num = sys.argv[3]
var0=[]
var1=[]
var2=[]
var3=[]
var4=[]
var5=[]
var6=[]
var7=[]
var8=[]
var9=[]
var10=[]
var11=[]
var12=[]

for n in range(int(num)):                                                
#data_3 = []
    FileName = "solution_" + sol + "_" + gen + "_" + str(n) +".npy"
    x= np.load(FileName)
    #print (x)
    var0.append(x[0])
    var1.append(x[1])
    var2.append(x[2])
    var3.append(x[3])
    var4.append(x[4])
    var5.append(x[5])
    var6.append(x[6])
    var7.append(x[7])
    var8.append(x[8])
    var9.append(x[9])
    var10.append(x[10])
    var11.append(x[11])
    var12.append(x[12])
    
mu1pt =np.round(np.average(var0),2)
mu2pt = np.round(np.average(var1),2)
kpt = np.round(np.average(var2),2)
jpsi_pt = np.round(np.average(var3),2)
jpsivtx_svprob = np.round(np.average(var4),4)
jpsivtx_cos2D = np.round(np.average(var5),4)
bvtx_svprob = np.round(np.average(var6),4)
bvtx_cos2D = np.round(np.average(var7),4)
Bpt_reco = np.round(np.average(var8),2)
Bmass = np.round(np.average(var9),3)
DR_jpsimu = np.round(np.average(var10),3)
ip3d_ns = np.round(np.average(var11),4)
ip3d_ps = np.round(np.average(var12),4)

'''
values = []
for i in range (len(x)):
    values.append(v)
'''
values = [mu1pt,mu2pt,kpt,jpsi_pt,jpsivtx_svprob,jpsivtx_cos2D,bvtx_svprob,bvtx_cos2D,Bpt_reco,Bmass,DR_jpsimu,ip3d_ns,ip3d_ps]

#values =[2.8, 2.7, 2.6, 5.7, 0.005, 13.9, 0.988]
#values =[3.64, 2.99, 2.48, 6.65, 0.0036, 0.9599, 0.0119, 0.9925, 12.89, 3.7, 0.776, -0.0058, 0.0093]
print (values)
np.save('average_%s_%s_0p75'%(sol,gen),values)

tree_sig = ROOT.RDataFrame("BTo3Mu",'/home/users/parolia/pre_selection/final_selection/new_weights/Sig.root')
tree_bkg = ROOT.RDataFrame("BTo3Mu",'/home/users/parolia/pre_selection/final_selection/new_weights/Bkg.root')
tree_tau = ROOT.RDataFrame("BTo3Mu",'/home/users/parolia/pre_selection/final_selection/new_weights/Sig_tau.root')
tree_mu = ROOT.RDataFrame("BTo3Mu",'/home/users/parolia/pre_selection/final_selection/new_weights/Sig_mu.root')

sig_num = tree_sig.Count();
bkg_num = tree_bkg.Count();
tau_num = tree_tau.Count();
mu_num = tree_mu.Count();
                                                                 
var = ['mu1pt','mu2pt','kpt','jpsi_pt','jpsivtx_svprob','jpsivtx_cos2D','bvtx_svprob','bvtx_cos2D','Bpt_reco','Bmass','DR_jpsimu','ip3d']
#var = ['mu1pt','mu2pt','kpt','jpsi_pt','jpsivtx_svprob','Bpt_reco','bvtx_cos2D']

#low = [0,0,0,0,0,0,0.975]
#high = [30,30,30,30,1,80,1.005]

low = [0,0,0,0,0,0.975,0,0.975,0,2,0,-0.03]
high = [30,30,30,30,1,1.005,1,1.005,80,8.0,2,0.03]
'''
for i in range(12):

    c = ROOT.TCanvas()
    c.cd()
      
    #h_sig = ROOT.TH1D("hsig","Distribution of %s_sig with cuts"%var[i], 50, low[i], high[i])
    #h_bkg = ROOT.TH1D("hbkg","Distribution of %s_bkg with cuts"%var[i], 50, low[i], high[i])
    #h_tau = ROOT.TH1D("htau","Distribution of %s_tau with cuts"%var[i], 50, low[i], high[i])
    #h_mu = ROOT.TH1D("hmu","Distribution of %s_muu with cuts"%var[i], 50, low[i], high[i])

   
    hs = tree_sig.Histo1D(("hsig", "Distribution of %s_sig with cuts"%var[i], 100, low[i], high[i]), '%s'%var[i], "norm_weight")
    hb = tree_bkg.Histo1D(("hbkg", "Distribution of %s_bkg with cuts"%var[i], 100, low[i], high[i]), '%s'%var[i],"norm_weight")
    ht = tree_tau.Histo1D(("htau", "Distribution of %s_tau with cuts"%var[i], 100, low[i], high[i]), '%s'%var[i],"norm_weight")
    hm = tree_mu.Histo1D(("hmu", "Distribution of %s_mu with cuts"%var[i], 100, low[i], high[i]), '%s'%var[i],"norm_weight")

    h_sig = hs.Clone()
    h_bkg = hb.Clone()
    h_tau = ht.Clone()
    h_mu = hm.Clone()

    
    for entrySig in range (0,sig_num.GetValue()):
        tree_sig.Define('weight_s',"norm_weight")
        h_sig.Fill(var[i],weight_s)

    for entryBkg in range (0,bkg_num.GetValue()):
        #weight_b = getattr(tree_bkg ,"norm_weight")
        tree_bkg.Define('weight_b',"norm_weight")
        h_bkg.Fill(var[i],weight_b)

    for entryTau in range (0,tau_num.GetValue()):
        #weight_t = getattr(tree_tau ,"norm_weight")
        tree_tau.Define('weight_t',"norm_weight")
        h_tau.Fill(var[i],weight_t)

    for entryMu in range (0,mu_num.GetValue()):
        weight_m = getattr(tree_mu ,"norm_weight")
        h_mu.Fill(var[i],weight_m)
    

    h_sig.SetLineColor(ROOT.kBlack)
    h_bkg.SetLineColor(ROOT.kRed)
    h_tau.SetLineColor(ROOT.kGreen)
    h_mu.SetLineColor(ROOT.kBlue)
             
    h_sig.SetTitle("%s Distribution"%var[i])
    h_sig.GetYaxis().SetTitle("Events")
    h_sig.GetXaxis().SetTitle("%s"%var[i])
    maxValHisto = h_sig.GetMaximum()
    h_sig.SetMaximum(1.5 * maxValHisto)
    h_bkg.Scale(h_sig.Integral()/h_bkg.Integral())
    h_tau.Scale(h_sig.Integral()/h_tau.Integral())
    h_mu.Scale(h_sig.Integral()/h_mu.Integral())
    h_sig.DrawCopy("h")
    h_bkg.DrawCopy("h,SAME")
    h_tau.DrawCopy("h,SAME")
    h_mu.DrawCopy("h,SAME")

    if (i == 11):
        cut1 = float(values[i])                                                 
        cut2 = float(values[i+1])                                               
        graph = TGraph(3)                                                      
        graph2 = TGraph(3)                                                     
        #set three points of the graph                                          
        graph.SetPoint(0, cut1, 0)                                              
        graph.SetPoint(1, cut1, maxValHisto)                                   
        graph.SetPoint(2, cut1, 30000)                                        
        graph2.SetPoint(0, cut2, 0)                                             
        graph2.SetPoint(1, cut2, maxValHisto)                                  
        graph2.SetPoint(2, cut2, 30000)                                        
        graph.SetLineColor(2)                                             
        graph2.SetLineColor(2)                                             
        graph.Draw("same")  
        graph2.Draw("same")  

    else:
        
        cut = float(values[i])                                                  
        graph = TGraph(3)                                  
        #set three points of the graph                           
        graph.SetPoint(0, cut, 0)                                               
        graph.SetPoint(1, cut, maxValHisto)                                    
        graph.SetPoint(2, cut, 30000)                         
        graph.SetLineColor(2)                                                   
        graph.Draw("same")
    #hsig = TH1D* 
    
    legend = TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(h_sig, "sig", "l")
    legend.AddEntry(h_bkg, "bkg", "l")
    legend.AddEntry(h_tau, "tau", "l")
    legend.AddEntry(h_mu, "muon", "l")
    legend.AddEntry(graph, "cut", "l")
    legend.Draw()
    
    c.SaveAs('%s_cuts.png'%var[i])

'''

def validation(var, cuts, tree_sig, tree_bkg, param_abs):
    
    #calculating the efficiency of cuts
    denom_sig = tree_sig.Count();
    denom_bkg = tree_bkg.Count();
    entries_sig = tree_sig.Filter('%s > %s'%(var[0],cuts[0]))
    entries_bkg = tree_bkg.Filter('%s > %s'%(var[0],cuts[0]))
    
    for i in range(1,(len(var)-(param_abs+1))):
        entries_sig = entries_sig.Filter('%s > %s'%(var[i],cuts[i]))
        entries_bkg = entries_bkg.Filter('%s > %s'%(var[i],cuts[i]))
    
    for k in range(param_abs):
        entries_sig = entries_sig.Filter('%s < %s'%(var[k+i+1],cuts[k+i+1]))
        entries_bkg = entries_bkg.Filter('%s < %s'%(var[k+i+1],cuts[k+i+1]))
    
    entries_sig = entries_sig.Filter('(%s > %s) and (%s < %s)'%(var[k+i+2],cuts[k+i+2],var[k+i+2],cuts[k+i+3]))
    entries_bkg = entries_bkg.Filter('(%s > %s) and (%s < %s)'%(var[k+i+2],cuts[k+i+2],var[k+i+2],cuts[k+i+3]))
    
    sig_ent = entries_sig.Count();
    bkg_ent = entries_bkg.Count();
    sig = sig_ent.GetValue()
    bkg = bkg_ent.GetValue()

    tot_tau = (tree_sig.Filter ("is_signal_channel > 0.5")).Count();            
    tot_muon = (tree_sig.Filter ("is_signal_channel < 0.5")).Count();           
    cut_tau = (entries_sig.Filter ("is_signal_channel > 0.5")).Count();         
    cut_muon = (entries_sig.Filter ("is_signal_channel < 0.5")).Count();       
    eff_tau = np.round(cut_tau.GetValue()/tot_tau.GetValue(),3)
    eff_muon = np.round(cut_muon.GetValue()/tot_muon.GetValue(),3)
    eff_sig = np.round(sig/(denom_sig.GetValue()),3)
    eff_bkg = np.round(bkg/(denom_bkg.GetValue()),3)
    
    return eff_sig, eff_bkg, eff_tau, eff_muon

param_abs = 1
(eff_sig,eff_bkg, eff_tau, eff_muon) = validation(var, values, tree_sig, tree_bkg, param_abs)

print (eff_sig , eff_bkg)
print (eff_tau , eff_muon)

