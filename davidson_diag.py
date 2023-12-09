import numpy as np
import scipy
from scipy import linalg
import time

N = 10000

# construct a real symmetric matrix
sparsity = 0.0001
A = np.zeros((N,N))
for i in range(N):
  A[i,i] = i + 1
A = A + sparsity*np.random.randn(N,N)
A = (A.T + A)/2

k = 4# number of eigenvalues
l = 6# l >= k
B = np.eye(N,l)
theta = np.zeros(N)
I = np.eye(N)

T1 = time.perf_counter()
for a in range(N//2):
  _A_ = np.einsum('ki,kl,lj->ij',B,A,B)# most expensive
  Value,Coe = linalg.eigh(_A_)# dim of subspace is l
  q = np.einsum('ij,jk,k->i',(A-Value[k-1]*I),B,Coe[:,k-1])
  norm_q = np.linalg.norm(q)
  if norm_q<1e-8 and l!=8 : break
  for i in range(N):
    theta[i] = q[i]/(Value[k-1]-A[i,i])
  B = np.insert(B,l,theta,axis=1)
  B,R = np.linalg.qr(B)
  l = l+1
print(Value[k-1],'\n',l-8)
T2 = time.perf_counter()

T3 = time.perf_counter()
AA ,AAA = linalg.eigh(A)
print(AA[k-1])
T4 = time.perf_counter()

print('davidson:',(T2-T1)*1000)
print('scipy   :',(T4-T3)*1000)
