library(mcmcse)
mh_biv = read.csv('mh_chains_biv.csv')
hm_biv = read.csv('hm_chains_biv.csv')
rej_biv = read.csv('rej_chains_biv.csv')
df_biv <- data.frame(chain=integer(),alg=character(),ess0 = double(),ess1=double(), stringsAsFactors = FALSE)
t = 5
for( i in 1:5){
  df_biv[i,] <- c(i,'mh',ess(mh_biv[mh_biv$chain_no == i,'x0']),ess(mh_biv[mh_biv$chain_no == i,'x1']))
  df_biv[i+t,] <- c(i,'hm',ess(hm_biv[hm_biv$chain_no == i,'x0']),ess(hm_biv[hm_biv$chain_no == i,'x1']))
  df_biv[i+2*t,] <- c(i,'rej',ess(rej_biv[rej_biv$chain_no == i,'x0']),ess(rej_biv[rej_biv$chain_no == i,'x1']))
  
}
write.csv(df_biv, row.names = FALSE)
mh_ban = read.csv('mh_chains_ban.csv')
hm_ban = read.csv('hm_chains_ban.csv')
rej_ban = read.csv('rej_chains_ban.csv')
df_ban <- data.frame(chain=integer(),alg=character(),ess0 = double(),ess1=double(), stringsAsFactors = FALSE)
t = 5
for( i in 1:5){
  df_ban[i,] <- c(i,'mh',ess(mh_ban[mh_ban$chain_no == i,'x0']),ess(mh_ban[mh_ban$chain_no == i,'x1']))
  df_ban[i+t,] <- c(i,'hm',ess(hm_ban[hm_ban$chain_no == i,'x0']),ess(hm_ban[hm_ban$chain_no == i,'x1']))
  df_ban[i+2*t,] <- c(i,'rej',ess(rej_ban[rej_ban$chain_no == i,'x0']),ess(rej_ban[rej_ban$chain_no == i,'x1']))
}

mh_lr = read.csv('mh_lr_s.csv')
hm_lr = read.csv('hm_lr_s.csv')
rej_lr = read.csv('rej_chains_lr.csv')

df_lr <- data.frame(chain=integer(),alg=character(),ess0 = double(),ess1=double(), stringsAsFactors = FALSE)
t = 5
for( i in 1:5){
  df_lr[i,] <- c(i,'mh',ess(mh_lr[mh_lr$chain_no == i,'x0']),ess(mh_lr[mh_lr$chain_no == i,'x1']))
  df_lr[i+t,] <- c(i,'hm',ess(hm_lr[hm_lr$chain_no == i,'x0']),ess(hm_lr[hm_lr$chain_no == i,'x1']))
  df_lr[i+2*t,] <- c(i,'rej',ess(rej_lr[rej_lr$chain_no == i,'x0']),ess(rej_lr[rej_lr$chain_no == i,'x1']))
}


mh_lr_full = read.csv('mh_lr_s_full.csv')
hm_lr_full = read.csv('hm_lr_s_full.csv')

df_lr_full <- data.frame(chain=integer(),alg=character(),ess0 = double(),ess1=double(),ess2=double(),ess3=double(),ess4=double(),ess5=double(),ess6=double(),ess7=double(),ess8=double(),ess9=double(),ess10=double(), stringsAsFactors = FALSE)
t = 5
for( i in 1:5){
  df_lr_full[i,] <- c(i,'mh',ess(mh_lr_full[mh_lr_full$chain_no == i,'x0']),ess(mh_lr_full[mh_lr_full$chain_no == i,'x1']),ess(mh_lr_full[mh_lr_full$chain_no == i,'x2']),ess(mh_lr_full[mh_lr_full$chain_no == i,'x3']),ess(mh_lr_full[mh_lr_full$chain_no == i,'x4']),ess(mh_lr_full[mh_lr_full$chain_no == i,'x5']),ess(mh_lr_full[mh_lr_full$chain_no == i,'x6']),ess(mh_lr_full[mh_lr_full$chain_no == i,'x7']),ess(mh_lr_full[mh_lr_full$chain_no == i,'x8']),ess(mh_lr_full[mh_lr_full$chain_no == i,'x9']),ess(mh_lr_full[mh_lr_full$chain_no == i,'x10']))
  df_lr_full[i+t,] <- c(i,'hm',ess(hm_lr_full[hm_lr_full$chain_no == i,'x0']),ess(hm_lr_full[hm_lr_full$chain_no == i,'x1']),ess(hm_lr_full[hm_lr_full$chain_no == i,'x2']),ess(hm_lr_full[hm_lr_full$chain_no == i,'x3']),ess(hm_lr_full[hm_lr_full$chain_no == i,'x4']),ess(hm_lr_full[hm_lr_full$chain_no == i,'x5']),ess(hm_lr_full[hm_lr_full$chain_no == i,'x6']),ess(hm_lr_full[hm_lr_full$chain_no == i,'x7']),ess(hm_lr_full[hm_lr_full$chain_no == i,'x8']),ess(hm_lr_full[hm_lr_full$chain_no == i,'x9']),ess(hm_lr_full[hm_lr_full$chain_no == i,'x10']))
  }
print('ess-biv:')
print('mh:')
print(ess(mh_biv$x0))
print(ess(mh_biv$x1))
print('hm:')
print(ess(hm_biv$x0))
print(ess(hm_biv$x1))
print('rej:')
print(ess(rej_biv$x0))
print(ess(rej_biv$x1))

print('ess-ban:')
print('mh:')
print(ess(mh_ban$x0))
print(ess(mh_ban$x1))
print('hm:')
print(ess(hm_ban$x0))
print(ess(hm_ban$x1))
print('rej:')
print(ess(rej_ban$x0))
print(ess(rej_ban$x1))

print('ess-lr-2:')
print('mh:')
print(ess(mh_lr$x0))
print(ess(mh_lr$x1))
print('hm:')
print(ess(hm_lr$x0))
print(ess(hm_lr$x1))
print('rej:')
print(ess(rej_lr$x0))
print(ess(rej_lr$x1))


print('ess-lr-full:')
print('mh:')
print(ess(mh_lr_full$x0))
print(ess(mh_lr_full$x1))
print('hm:')
print(ess(hm_lr_full$x0))
print(ess(hm_lr_full$x1))

