щи
ш╛
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12v2.4.1-0-g85c8b2a817f8ю╕	
Д
conv2d_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_39/kernel
}
$conv2d_39/kernel/Read/ReadVariableOpReadVariableOpconv2d_39/kernel*&
_output_shapes
: *
dtype0
t
conv2d_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_39/bias
m
"conv2d_39/bias/Read/ReadVariableOpReadVariableOpconv2d_39/bias*
_output_shapes
: *
dtype0
Д
conv2d_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_40/kernel
}
$conv2d_40/kernel/Read/ReadVariableOpReadVariableOpconv2d_40/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_40/bias
m
"conv2d_40/bias/Read/ReadVariableOpReadVariableOpconv2d_40/bias*
_output_shapes
:@*
dtype0
Е
conv2d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*!
shared_nameconv2d_41/kernel
~
$conv2d_41/kernel/Read/ReadVariableOpReadVariableOpconv2d_41/kernel*'
_output_shapes
:@А*
dtype0
u
conv2d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_41/bias
n
"conv2d_41/bias/Read/ReadVariableOpReadVariableOpconv2d_41/bias*
_output_shapes	
:А*
dtype0
{
dense_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Аdx* 
shared_namedense_39/kernel
t
#dense_39/kernel/Read/ReadVariableOpReadVariableOpdense_39/kernel*
_output_shapes
:	Аdx*
dtype0
r
dense_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*
shared_namedense_39/bias
k
!dense_39/bias/Read/ReadVariableOpReadVariableOpdense_39/bias*
_output_shapes
:x*
dtype0
z
dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x<* 
shared_namedense_40/kernel
s
#dense_40/kernel/Read/ReadVariableOpReadVariableOpdense_40/kernel*
_output_shapes

:x<*
dtype0
r
dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*
shared_namedense_40/bias
k
!dense_40/bias/Read/ReadVariableOpReadVariableOpdense_40/bias*
_output_shapes
:<*
dtype0
z
dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<* 
shared_namedense_41/kernel
s
#dense_41/kernel/Read/ReadVariableOpReadVariableOpdense_41/kernel*
_output_shapes

:<*
dtype0
r
dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_41/bias
k
!dense_41/bias/Read/ReadVariableOpReadVariableOpdense_41/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
Т
Adam/conv2d_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_39/kernel/m
Л
+Adam/conv2d_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/kernel/m*&
_output_shapes
: *
dtype0
В
Adam/conv2d_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_39/bias/m
{
)Adam/conv2d_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/bias/m*
_output_shapes
: *
dtype0
Т
Adam/conv2d_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_40/kernel/m
Л
+Adam/conv2d_40/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/kernel/m*&
_output_shapes
: @*
dtype0
В
Adam/conv2d_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_40/bias/m
{
)Adam/conv2d_40/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/bias/m*
_output_shapes
:@*
dtype0
У
Adam/conv2d_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*(
shared_nameAdam/conv2d_41/kernel/m
М
+Adam/conv2d_41/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/kernel/m*'
_output_shapes
:@А*
dtype0
Г
Adam/conv2d_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_41/bias/m
|
)Adam/conv2d_41/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/bias/m*
_output_shapes	
:А*
dtype0
Й
Adam/dense_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Аdx*'
shared_nameAdam/dense_39/kernel/m
В
*Adam/dense_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/m*
_output_shapes
:	Аdx*
dtype0
А
Adam/dense_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*%
shared_nameAdam/dense_39/bias/m
y
(Adam/dense_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/m*
_output_shapes
:x*
dtype0
И
Adam/dense_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x<*'
shared_nameAdam/dense_40/kernel/m
Б
*Adam/dense_40/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_40/kernel/m*
_output_shapes

:x<*
dtype0
А
Adam/dense_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*%
shared_nameAdam/dense_40/bias/m
y
(Adam/dense_40/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_40/bias/m*
_output_shapes
:<*
dtype0
И
Adam/dense_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*'
shared_nameAdam/dense_41/kernel/m
Б
*Adam/dense_41/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_41/kernel/m*
_output_shapes

:<*
dtype0
А
Adam/dense_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_41/bias/m
y
(Adam/dense_41/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_41/bias/m*
_output_shapes
:*
dtype0
Т
Adam/conv2d_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_39/kernel/v
Л
+Adam/conv2d_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/kernel/v*&
_output_shapes
: *
dtype0
В
Adam/conv2d_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_39/bias/v
{
)Adam/conv2d_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/bias/v*
_output_shapes
: *
dtype0
Т
Adam/conv2d_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_40/kernel/v
Л
+Adam/conv2d_40/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/kernel/v*&
_output_shapes
: @*
dtype0
В
Adam/conv2d_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_40/bias/v
{
)Adam/conv2d_40/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/bias/v*
_output_shapes
:@*
dtype0
У
Adam/conv2d_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*(
shared_nameAdam/conv2d_41/kernel/v
М
+Adam/conv2d_41/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/kernel/v*'
_output_shapes
:@А*
dtype0
Г
Adam/conv2d_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_41/bias/v
|
)Adam/conv2d_41/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/bias/v*
_output_shapes	
:А*
dtype0
Й
Adam/dense_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Аdx*'
shared_nameAdam/dense_39/kernel/v
В
*Adam/dense_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/v*
_output_shapes
:	Аdx*
dtype0
А
Adam/dense_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*%
shared_nameAdam/dense_39/bias/v
y
(Adam/dense_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/v*
_output_shapes
:x*
dtype0
И
Adam/dense_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x<*'
shared_nameAdam/dense_40/kernel/v
Б
*Adam/dense_40/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_40/kernel/v*
_output_shapes

:x<*
dtype0
А
Adam/dense_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*%
shared_nameAdam/dense_40/bias/v
y
(Adam/dense_40/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_40/bias/v*
_output_shapes
:<*
dtype0
И
Adam/dense_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*'
shared_nameAdam/dense_41/kernel/v
Б
*Adam/dense_41/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_41/kernel/v*
_output_shapes

:<*
dtype0
А
Adam/dense_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_41/bias/v
y
(Adam/dense_41/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_41/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
╓K
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*СK
valueЗKBДK B¤J
Э
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
R
"	variables
#regularization_losses
$trainable_variables
%	keras_api
h

&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
R
,	variables
-regularization_losses
.trainable_variables
/	keras_api
R
0	variables
1regularization_losses
2trainable_variables
3	keras_api
h

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
h

:kernel
;bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
R
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
h

Dkernel
Ebias
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
░
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratemЦmЧmШmЩ&mЪ'mЫ4mЬ5mЭ:mЮ;mЯDmаEmбvвvгvдvе&vж'vз4vи5vй:vк;vлDvмEvн
V
0
1
2
3
&4
'5
46
57
:8
;9
D10
E11
 
V
0
1
2
3
&4
'5
46
57
:8
;9
D10
E11
н
	variables

Olayers
Pmetrics
regularization_losses
trainable_variables
Qlayer_regularization_losses
Rlayer_metrics
Snon_trainable_variables
 
\Z
VARIABLE_VALUEconv2d_39/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_39/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
н
	variables

Tlayers
Umetrics
regularization_losses
trainable_variables
Vlayer_regularization_losses
Wlayer_metrics
Xnon_trainable_variables
 
 
 
н
	variables

Ylayers
Zmetrics
regularization_losses
trainable_variables
[layer_regularization_losses
\layer_metrics
]non_trainable_variables
\Z
VARIABLE_VALUEconv2d_40/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_40/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
н
	variables

^layers
_metrics
regularization_losses
 trainable_variables
`layer_regularization_losses
alayer_metrics
bnon_trainable_variables
 
 
 
н
"	variables

clayers
dmetrics
#regularization_losses
$trainable_variables
elayer_regularization_losses
flayer_metrics
gnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_41/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_41/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1
 

&0
'1
н
(	variables

hlayers
imetrics
)regularization_losses
*trainable_variables
jlayer_regularization_losses
klayer_metrics
lnon_trainable_variables
 
 
 
н
,	variables

mlayers
nmetrics
-regularization_losses
.trainable_variables
olayer_regularization_losses
player_metrics
qnon_trainable_variables
 
 
 
н
0	variables

rlayers
smetrics
1regularization_losses
2trainable_variables
tlayer_regularization_losses
ulayer_metrics
vnon_trainable_variables
[Y
VARIABLE_VALUEdense_39/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_39/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51
н
6	variables

wlayers
xmetrics
7regularization_losses
8trainable_variables
ylayer_regularization_losses
zlayer_metrics
{non_trainable_variables
[Y
VARIABLE_VALUEdense_40/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_40/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
 

:0
;1
о
<	variables

|layers
}metrics
=regularization_losses
>trainable_variables
~layer_regularization_losses
layer_metrics
Аnon_trainable_variables
 
 
 
▓
@	variables
Бlayers
Вmetrics
Aregularization_losses
Btrainable_variables
 Гlayer_regularization_losses
Дlayer_metrics
Еnon_trainable_variables
[Y
VARIABLE_VALUEdense_41/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_41/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1
 

D0
E1
▓
F	variables
Жlayers
Зmetrics
Gregularization_losses
Htrainable_variables
 Иlayer_regularization_losses
Йlayer_metrics
Кnon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
N
0
1
2
3
4
5
6
7
	8

9
10

Л0
М1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

Нtotal

Оcount
П	variables
Р	keras_api
I

Сtotal

Тcount
У
_fn_kwargs
Ф	variables
Х	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Н0
О1

П	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

С0
Т1

Ф	variables
}
VARIABLE_VALUEAdam/conv2d_39/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_39/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_40/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_40/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_41/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_41/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_39/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_39/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_40/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_40/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_41/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_41/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_39/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_39/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_40/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_40/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_41/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_41/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_39/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_39/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_40/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_40/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_41/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_41/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ц
serving_default_conv2d_39_inputPlaceholder*1
_output_shapes
:         АА*
dtype0*&
shape:         АА
Щ
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_39_inputconv2d_39/kernelconv2d_39/biasconv2d_40/kernelconv2d_40/biasconv2d_41/kernelconv2d_41/biasdense_39/kerneldense_39/biasdense_40/kerneldense_40/biasdense_41/kerneldense_41/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference_signature_wrapper_5756
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
м
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_39/kernel/Read/ReadVariableOp"conv2d_39/bias/Read/ReadVariableOp$conv2d_40/kernel/Read/ReadVariableOp"conv2d_40/bias/Read/ReadVariableOp$conv2d_41/kernel/Read/ReadVariableOp"conv2d_41/bias/Read/ReadVariableOp#dense_39/kernel/Read/ReadVariableOp!dense_39/bias/Read/ReadVariableOp#dense_40/kernel/Read/ReadVariableOp!dense_40/bias/Read/ReadVariableOp#dense_41/kernel/Read/ReadVariableOp!dense_41/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_39/kernel/m/Read/ReadVariableOp)Adam/conv2d_39/bias/m/Read/ReadVariableOp+Adam/conv2d_40/kernel/m/Read/ReadVariableOp)Adam/conv2d_40/bias/m/Read/ReadVariableOp+Adam/conv2d_41/kernel/m/Read/ReadVariableOp)Adam/conv2d_41/bias/m/Read/ReadVariableOp*Adam/dense_39/kernel/m/Read/ReadVariableOp(Adam/dense_39/bias/m/Read/ReadVariableOp*Adam/dense_40/kernel/m/Read/ReadVariableOp(Adam/dense_40/bias/m/Read/ReadVariableOp*Adam/dense_41/kernel/m/Read/ReadVariableOp(Adam/dense_41/bias/m/Read/ReadVariableOp+Adam/conv2d_39/kernel/v/Read/ReadVariableOp)Adam/conv2d_39/bias/v/Read/ReadVariableOp+Adam/conv2d_40/kernel/v/Read/ReadVariableOp)Adam/conv2d_40/bias/v/Read/ReadVariableOp+Adam/conv2d_41/kernel/v/Read/ReadVariableOp)Adam/conv2d_41/bias/v/Read/ReadVariableOp*Adam/dense_39/kernel/v/Read/ReadVariableOp(Adam/dense_39/bias/v/Read/ReadVariableOp*Adam/dense_40/kernel/v/Read/ReadVariableOp(Adam/dense_40/bias/v/Read/ReadVariableOp*Adam/dense_41/kernel/v/Read/ReadVariableOp(Adam/dense_41/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *&
f!R
__inference__traced_save_6241
г	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_39/kernelconv2d_39/biasconv2d_40/kernelconv2d_40/biasconv2d_41/kernelconv2d_41/biasdense_39/kerneldense_39/biasdense_40/kerneldense_40/biasdense_41/kerneldense_41/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_39/kernel/mAdam/conv2d_39/bias/mAdam/conv2d_40/kernel/mAdam/conv2d_40/bias/mAdam/conv2d_41/kernel/mAdam/conv2d_41/bias/mAdam/dense_39/kernel/mAdam/dense_39/bias/mAdam/dense_40/kernel/mAdam/dense_40/bias/mAdam/dense_41/kernel/mAdam/dense_41/bias/mAdam/conv2d_39/kernel/vAdam/conv2d_39/bias/vAdam/conv2d_40/kernel/vAdam/conv2d_40/bias/vAdam/conv2d_41/kernel/vAdam/conv2d_41/bias/vAdam/dense_39/kernel/vAdam/dense_39/bias/vAdam/dense_40/kernel/vAdam/dense_40/bias/vAdam/dense_41/kernel/vAdam/dense_41/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__traced_restore_6386УЎ
о
K
/__inference_max_pooling2d_41_layer_call_fn_5327

inputs
identityю
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_53212
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
а0
е
G__inference_sequential_13_layer_call_and_return_conditional_losses_5690

inputs
conv2d_39_5654
conv2d_39_5656
conv2d_40_5660
conv2d_40_5662
conv2d_41_5666
conv2d_41_5668
dense_39_5673
dense_39_5675
dense_40_5678
dense_40_5680
dense_41_5684
dense_41_5686
identityИв!conv2d_39/StatefulPartitionedCallв!conv2d_40/StatefulPartitionedCallв!conv2d_41/StatefulPartitionedCallв dense_39/StatefulPartitionedCallв dense_40/StatefulPartitionedCallв dense_41/StatefulPartitionedCallа
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_39_5654conv2d_39_5656*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_39_layer_call_and_return_conditional_losses_53422#
!conv2d_39/StatefulPartitionedCallЩ
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_52972"
 max_pooling2d_39/PartitionedCall┴
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0conv2d_40_5660conv2d_40_5662*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_40_layer_call_and_return_conditional_losses_53702#
!conv2d_40/StatefulPartitionedCallЩ
 max_pooling2d_40/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_53092"
 max_pooling2d_40/PartitionedCall┬
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0conv2d_41_5666conv2d_41_5668*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_41_layer_call_and_return_conditional_losses_53982#
!conv2d_41/StatefulPartitionedCallЪ
 max_pooling2d_41/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         

А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_53212"
 max_pooling2d_41/PartitionedCall 
flatten_13/PartitionedCallPartitionedCall)max_pooling2d_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Аd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_13_layer_call_and_return_conditional_losses_54212
flatten_13/PartitionedCallо
 dense_39/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_39_5673dense_39_5675*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_39_layer_call_and_return_conditional_losses_54402"
 dense_39/StatefulPartitionedCall┤
 dense_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0dense_40_5678dense_40_5680*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_40_layer_call_and_return_conditional_losses_54672"
 dense_40/StatefulPartitionedCall■
dropout_13/PartitionedCallPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         <* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_13_layer_call_and_return_conditional_losses_55002
dropout_13/PartitionedCallо
 dense_41/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_41_5684dense_41_5686*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_41_layer_call_and_return_conditional_losses_55242"
 dense_41/StatefulPartitionedCall╥
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0"^conv2d_39/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         АА::::::::::::2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
о
K
/__inference_max_pooling2d_40_layer_call_fn_5315

inputs
identityю
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_53092
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
яC
ж
G__inference_sequential_13_layer_call_and_return_conditional_losses_5867

inputs,
(conv2d_39_conv2d_readvariableop_resource-
)conv2d_39_biasadd_readvariableop_resource,
(conv2d_40_conv2d_readvariableop_resource-
)conv2d_40_biasadd_readvariableop_resource,
(conv2d_41_conv2d_readvariableop_resource-
)conv2d_41_biasadd_readvariableop_resource+
'dense_39_matmul_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource+
'dense_40_matmul_readvariableop_resource,
(dense_40_biasadd_readvariableop_resource+
'dense_41_matmul_readvariableop_resource,
(dense_41_biasadd_readvariableop_resource
identityИв conv2d_39/BiasAdd/ReadVariableOpвconv2d_39/Conv2D/ReadVariableOpв conv2d_40/BiasAdd/ReadVariableOpвconv2d_40/Conv2D/ReadVariableOpв conv2d_41/BiasAdd/ReadVariableOpвconv2d_41/Conv2D/ReadVariableOpвdense_39/BiasAdd/ReadVariableOpвdense_39/MatMul/ReadVariableOpвdense_40/BiasAdd/ReadVariableOpвdense_40/MatMul/ReadVariableOpвdense_41/BiasAdd/ReadVariableOpвdense_41/MatMul/ReadVariableOp│
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_39/Conv2D/ReadVariableOp├
conv2d_39/Conv2DConv2Dinputs'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         АА *
paddingSAME*
strides
2
conv2d_39/Conv2Dк
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp▓
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         АА 2
conv2d_39/BiasAddА
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*1
_output_shapes
:         АА 2
conv2d_39/Relu╩
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*/
_output_shapes
:         @@ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPool│
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_40/Conv2D/ReadVariableOp▄
conv2d_40/Conv2DConv2D!max_pooling2d_39/MaxPool:output:0'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@@*
paddingSAME*
strides
2
conv2d_40/Conv2Dк
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp░
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@@2
conv2d_40/BiasAdd~
conv2d_40/ReluReluconv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:         @@@2
conv2d_40/Relu╩
max_pooling2d_40/MaxPoolMaxPoolconv2d_40/Relu:activations:0*/
_output_shapes
:           @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_40/MaxPool┤
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02!
conv2d_41/Conv2D/ReadVariableOp▌
conv2d_41/Conv2DConv2D!max_pooling2d_40/MaxPool:output:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2
conv2d_41/Conv2Dл
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp▒
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
conv2d_41/BiasAdd
conv2d_41/ReluReluconv2d_41/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
conv2d_41/Relu╦
max_pooling2d_41/MaxPoolMaxPoolconv2d_41/Relu:activations:0*0
_output_shapes
:         

А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_41/MaxPoolu
flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"     2  2
flatten_13/Constд
flatten_13/ReshapeReshape!max_pooling2d_41/MaxPool:output:0flatten_13/Const:output:0*
T0*(
_output_shapes
:         Аd2
flatten_13/Reshapeй
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	Аdx*
dtype02 
dense_39/MatMul/ReadVariableOpг
dense_39/MatMulMatMulflatten_13/Reshape:output:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
dense_39/MatMulз
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02!
dense_39/BiasAdd/ReadVariableOpе
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
dense_39/BiasAdds
dense_39/ReluReludense_39/BiasAdd:output:0*
T0*'
_output_shapes
:         x2
dense_39/Reluи
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:x<*
dtype02 
dense_40/MatMul/ReadVariableOpг
dense_40/MatMulMatMuldense_39/Relu:activations:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         <2
dense_40/MatMulз
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02!
dense_40/BiasAdd/ReadVariableOpе
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         <2
dense_40/BiasAdds
dense_40/ReluReludense_40/BiasAdd:output:0*
T0*'
_output_shapes
:         <2
dense_40/ReluЕ
dropout_13/IdentityIdentitydense_40/Relu:activations:0*
T0*'
_output_shapes
:         <2
dropout_13/Identityи
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02 
dense_41/MatMul/ReadVariableOpд
dense_41/MatMulMatMuldropout_13/Identity:output:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_41/MatMulз
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_41/BiasAdd/ReadVariableOpе
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_41/BiasAdd|
dense_41/SoftmaxSoftmaxdense_41/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_41/SoftmaxЖ
IdentityIdentitydense_41/Softmax:softmax:0!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         АА::::::::::::2D
 conv2d_39/BiasAdd/ReadVariableOp conv2d_39/BiasAdd/ReadVariableOp2B
conv2d_39/Conv2D/ReadVariableOpconv2d_39/Conv2D/ReadVariableOp2D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
┘

▄
C__inference_conv2d_39_layer_call_and_return_conditional_losses_5936

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpе
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         АА *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         АА 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         АА 2
Reluб
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:         АА 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         АА::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
я	
█
B__inference_dense_39_layer_call_and_return_conditional_losses_6007

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Аdx*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         x2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         x2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Аd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Аd
 
_user_specified_nameinputs
 
}
(__inference_conv2d_41_layer_call_fn_5985

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_41_layer_call_and_return_conditional_losses_53982
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:           @::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           @
 
_user_specified_nameinputs
╛
`
D__inference_flatten_13_layer_call_and_return_conditional_losses_5991

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     2  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         Аd2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         Аd2

Identity"
identityIdentity:output:0*/
_input_shapes
:         

А:X T
0
_output_shapes
:         

А
 
_user_specified_nameinputs
Ї	
█
B__inference_dense_41_layer_call_and_return_conditional_losses_6074

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         <::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         <
 
_user_specified_nameinputs
ь	
█
B__inference_dense_40_layer_call_and_return_conditional_losses_6027

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         <2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         <2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         <2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         <2

Identity"
identityIdentity:output:0*.
_input_shapes
:         x::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         x
 
_user_specified_nameinputs
╟
b
D__inference_dropout_13_layer_call_and_return_conditional_losses_6053

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         <2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         <2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         <:O K
'
_output_shapes
:         <
 
_user_specified_nameinputs
▌
|
'__inference_dense_39_layer_call_fn_6016

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_39_layer_call_and_return_conditional_losses_54402
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         x2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Аd::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Аd
 
_user_specified_nameinputs
Х
E
)__inference_dropout_13_layer_call_fn_6063

inputs
identity┼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         <* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_13_layer_call_and_return_conditional_losses_55002
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         <2

Identity"
identityIdentity:output:0*&
_input_shapes
:         <:O K
'
_output_shapes
:         <
 
_user_specified_nameinputs
А
f
J__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_5321

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┘

▄
C__inference_conv2d_39_layer_call_and_return_conditional_losses_5342

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpе
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         АА *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         АА 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         АА 2
Reluб
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:         АА 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         АА::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
╫	
б
,__inference_sequential_13_layer_call_fn_5717
conv2d_39_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallconv2d_39_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_13_layer_call_and_return_conditional_losses_56902
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         АА::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:         АА
)
_user_specified_nameconv2d_39_input
б
b
)__inference_dropout_13_layer_call_fn_6058

inputs
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         <* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_13_layer_call_and_return_conditional_losses_54952
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         <2

Identity"
identityIdentity:output:0*&
_input_shapes
:         <22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         <
 
_user_specified_nameinputs
└W
╫

__inference__wrapped_model_5291
conv2d_39_input:
6sequential_13_conv2d_39_conv2d_readvariableop_resource;
7sequential_13_conv2d_39_biasadd_readvariableop_resource:
6sequential_13_conv2d_40_conv2d_readvariableop_resource;
7sequential_13_conv2d_40_biasadd_readvariableop_resource:
6sequential_13_conv2d_41_conv2d_readvariableop_resource;
7sequential_13_conv2d_41_biasadd_readvariableop_resource9
5sequential_13_dense_39_matmul_readvariableop_resource:
6sequential_13_dense_39_biasadd_readvariableop_resource9
5sequential_13_dense_40_matmul_readvariableop_resource:
6sequential_13_dense_40_biasadd_readvariableop_resource9
5sequential_13_dense_41_matmul_readvariableop_resource:
6sequential_13_dense_41_biasadd_readvariableop_resource
identityИв.sequential_13/conv2d_39/BiasAdd/ReadVariableOpв-sequential_13/conv2d_39/Conv2D/ReadVariableOpв.sequential_13/conv2d_40/BiasAdd/ReadVariableOpв-sequential_13/conv2d_40/Conv2D/ReadVariableOpв.sequential_13/conv2d_41/BiasAdd/ReadVariableOpв-sequential_13/conv2d_41/Conv2D/ReadVariableOpв-sequential_13/dense_39/BiasAdd/ReadVariableOpв,sequential_13/dense_39/MatMul/ReadVariableOpв-sequential_13/dense_40/BiasAdd/ReadVariableOpв,sequential_13/dense_40/MatMul/ReadVariableOpв-sequential_13/dense_41/BiasAdd/ReadVariableOpв,sequential_13/dense_41/MatMul/ReadVariableOp▌
-sequential_13/conv2d_39/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_39_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_13/conv2d_39/Conv2D/ReadVariableOpЎ
sequential_13/conv2d_39/Conv2DConv2Dconv2d_39_input5sequential_13/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         АА *
paddingSAME*
strides
2 
sequential_13/conv2d_39/Conv2D╘
.sequential_13/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_39_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_13/conv2d_39/BiasAdd/ReadVariableOpъ
sequential_13/conv2d_39/BiasAddBiasAdd'sequential_13/conv2d_39/Conv2D:output:06sequential_13/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         АА 2!
sequential_13/conv2d_39/BiasAddк
sequential_13/conv2d_39/ReluRelu(sequential_13/conv2d_39/BiasAdd:output:0*
T0*1
_output_shapes
:         АА 2
sequential_13/conv2d_39/ReluЇ
&sequential_13/max_pooling2d_39/MaxPoolMaxPool*sequential_13/conv2d_39/Relu:activations:0*/
_output_shapes
:         @@ *
ksize
*
paddingVALID*
strides
2(
&sequential_13/max_pooling2d_39/MaxPool▌
-sequential_13/conv2d_40/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02/
-sequential_13/conv2d_40/Conv2D/ReadVariableOpФ
sequential_13/conv2d_40/Conv2DConv2D/sequential_13/max_pooling2d_39/MaxPool:output:05sequential_13/conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@@*
paddingSAME*
strides
2 
sequential_13/conv2d_40/Conv2D╘
.sequential_13/conv2d_40/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_13/conv2d_40/BiasAdd/ReadVariableOpш
sequential_13/conv2d_40/BiasAddBiasAdd'sequential_13/conv2d_40/Conv2D:output:06sequential_13/conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@@2!
sequential_13/conv2d_40/BiasAddи
sequential_13/conv2d_40/ReluRelu(sequential_13/conv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:         @@@2
sequential_13/conv2d_40/ReluЇ
&sequential_13/max_pooling2d_40/MaxPoolMaxPool*sequential_13/conv2d_40/Relu:activations:0*/
_output_shapes
:           @*
ksize
*
paddingVALID*
strides
2(
&sequential_13/max_pooling2d_40/MaxPool▐
-sequential_13/conv2d_41/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_41_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02/
-sequential_13/conv2d_41/Conv2D/ReadVariableOpХ
sequential_13/conv2d_41/Conv2DConv2D/sequential_13/max_pooling2d_40/MaxPool:output:05sequential_13/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2 
sequential_13/conv2d_41/Conv2D╒
.sequential_13/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_13/conv2d_41/BiasAdd/ReadVariableOpщ
sequential_13/conv2d_41/BiasAddBiasAdd'sequential_13/conv2d_41/Conv2D:output:06sequential_13/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2!
sequential_13/conv2d_41/BiasAddй
sequential_13/conv2d_41/ReluRelu(sequential_13/conv2d_41/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
sequential_13/conv2d_41/Reluї
&sequential_13/max_pooling2d_41/MaxPoolMaxPool*sequential_13/conv2d_41/Relu:activations:0*0
_output_shapes
:         

А*
ksize
*
paddingVALID*
strides
2(
&sequential_13/max_pooling2d_41/MaxPoolС
sequential_13/flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"     2  2 
sequential_13/flatten_13/Const▄
 sequential_13/flatten_13/ReshapeReshape/sequential_13/max_pooling2d_41/MaxPool:output:0'sequential_13/flatten_13/Const:output:0*
T0*(
_output_shapes
:         Аd2"
 sequential_13/flatten_13/Reshape╙
,sequential_13/dense_39/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_39_matmul_readvariableop_resource*
_output_shapes
:	Аdx*
dtype02.
,sequential_13/dense_39/MatMul/ReadVariableOp█
sequential_13/dense_39/MatMulMatMul)sequential_13/flatten_13/Reshape:output:04sequential_13/dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
sequential_13/dense_39/MatMul╤
-sequential_13/dense_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_39_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02/
-sequential_13/dense_39/BiasAdd/ReadVariableOp▌
sequential_13/dense_39/BiasAddBiasAdd'sequential_13/dense_39/MatMul:product:05sequential_13/dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2 
sequential_13/dense_39/BiasAddЭ
sequential_13/dense_39/ReluRelu'sequential_13/dense_39/BiasAdd:output:0*
T0*'
_output_shapes
:         x2
sequential_13/dense_39/Relu╥
,sequential_13/dense_40/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_40_matmul_readvariableop_resource*
_output_shapes

:x<*
dtype02.
,sequential_13/dense_40/MatMul/ReadVariableOp█
sequential_13/dense_40/MatMulMatMul)sequential_13/dense_39/Relu:activations:04sequential_13/dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         <2
sequential_13/dense_40/MatMul╤
-sequential_13/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_40_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02/
-sequential_13/dense_40/BiasAdd/ReadVariableOp▌
sequential_13/dense_40/BiasAddBiasAdd'sequential_13/dense_40/MatMul:product:05sequential_13/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         <2 
sequential_13/dense_40/BiasAddЭ
sequential_13/dense_40/ReluRelu'sequential_13/dense_40/BiasAdd:output:0*
T0*'
_output_shapes
:         <2
sequential_13/dense_40/Reluп
!sequential_13/dropout_13/IdentityIdentity)sequential_13/dense_40/Relu:activations:0*
T0*'
_output_shapes
:         <2#
!sequential_13/dropout_13/Identity╥
,sequential_13/dense_41/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_41_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02.
,sequential_13/dense_41/MatMul/ReadVariableOp▄
sequential_13/dense_41/MatMulMatMul*sequential_13/dropout_13/Identity:output:04sequential_13/dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_13/dense_41/MatMul╤
-sequential_13/dense_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_41_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_13/dense_41/BiasAdd/ReadVariableOp▌
sequential_13/dense_41/BiasAddBiasAdd'sequential_13/dense_41/MatMul:product:05sequential_13/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2 
sequential_13/dense_41/BiasAddж
sequential_13/dense_41/SoftmaxSoftmax'sequential_13/dense_41/BiasAdd:output:0*
T0*'
_output_shapes
:         2 
sequential_13/dense_41/Softmax╝
IdentityIdentity(sequential_13/dense_41/Softmax:softmax:0/^sequential_13/conv2d_39/BiasAdd/ReadVariableOp.^sequential_13/conv2d_39/Conv2D/ReadVariableOp/^sequential_13/conv2d_40/BiasAdd/ReadVariableOp.^sequential_13/conv2d_40/Conv2D/ReadVariableOp/^sequential_13/conv2d_41/BiasAdd/ReadVariableOp.^sequential_13/conv2d_41/Conv2D/ReadVariableOp.^sequential_13/dense_39/BiasAdd/ReadVariableOp-^sequential_13/dense_39/MatMul/ReadVariableOp.^sequential_13/dense_40/BiasAdd/ReadVariableOp-^sequential_13/dense_40/MatMul/ReadVariableOp.^sequential_13/dense_41/BiasAdd/ReadVariableOp-^sequential_13/dense_41/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         АА::::::::::::2`
.sequential_13/conv2d_39/BiasAdd/ReadVariableOp.sequential_13/conv2d_39/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_39/Conv2D/ReadVariableOp-sequential_13/conv2d_39/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_40/BiasAdd/ReadVariableOp.sequential_13/conv2d_40/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_40/Conv2D/ReadVariableOp-sequential_13/conv2d_40/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_41/BiasAdd/ReadVariableOp.sequential_13/conv2d_41/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_41/Conv2D/ReadVariableOp-sequential_13/conv2d_41/Conv2D/ReadVariableOp2^
-sequential_13/dense_39/BiasAdd/ReadVariableOp-sequential_13/dense_39/BiasAdd/ReadVariableOp2\
,sequential_13/dense_39/MatMul/ReadVariableOp,sequential_13/dense_39/MatMul/ReadVariableOp2^
-sequential_13/dense_40/BiasAdd/ReadVariableOp-sequential_13/dense_40/BiasAdd/ReadVariableOp2\
,sequential_13/dense_40/MatMul/ReadVariableOp,sequential_13/dense_40/MatMul/ReadVariableOp2^
-sequential_13/dense_41/BiasAdd/ReadVariableOp-sequential_13/dense_41/BiasAdd/ReadVariableOp2\
,sequential_13/dense_41/MatMul/ReadVariableOp,sequential_13/dense_41/MatMul/ReadVariableOp:b ^
1
_output_shapes
:         АА
)
_user_specified_nameconv2d_39_input
оM
ж
G__inference_sequential_13_layer_call_and_return_conditional_losses_5815

inputs,
(conv2d_39_conv2d_readvariableop_resource-
)conv2d_39_biasadd_readvariableop_resource,
(conv2d_40_conv2d_readvariableop_resource-
)conv2d_40_biasadd_readvariableop_resource,
(conv2d_41_conv2d_readvariableop_resource-
)conv2d_41_biasadd_readvariableop_resource+
'dense_39_matmul_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource+
'dense_40_matmul_readvariableop_resource,
(dense_40_biasadd_readvariableop_resource+
'dense_41_matmul_readvariableop_resource,
(dense_41_biasadd_readvariableop_resource
identityИв conv2d_39/BiasAdd/ReadVariableOpвconv2d_39/Conv2D/ReadVariableOpв conv2d_40/BiasAdd/ReadVariableOpвconv2d_40/Conv2D/ReadVariableOpв conv2d_41/BiasAdd/ReadVariableOpвconv2d_41/Conv2D/ReadVariableOpвdense_39/BiasAdd/ReadVariableOpвdense_39/MatMul/ReadVariableOpвdense_40/BiasAdd/ReadVariableOpвdense_40/MatMul/ReadVariableOpвdense_41/BiasAdd/ReadVariableOpвdense_41/MatMul/ReadVariableOp│
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_39/Conv2D/ReadVariableOp├
conv2d_39/Conv2DConv2Dinputs'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         АА *
paddingSAME*
strides
2
conv2d_39/Conv2Dк
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp▓
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         АА 2
conv2d_39/BiasAddА
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*1
_output_shapes
:         АА 2
conv2d_39/Relu╩
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*/
_output_shapes
:         @@ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPool│
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_40/Conv2D/ReadVariableOp▄
conv2d_40/Conv2DConv2D!max_pooling2d_39/MaxPool:output:0'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@@*
paddingSAME*
strides
2
conv2d_40/Conv2Dк
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp░
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@@2
conv2d_40/BiasAdd~
conv2d_40/ReluReluconv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:         @@@2
conv2d_40/Relu╩
max_pooling2d_40/MaxPoolMaxPoolconv2d_40/Relu:activations:0*/
_output_shapes
:           @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_40/MaxPool┤
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02!
conv2d_41/Conv2D/ReadVariableOp▌
conv2d_41/Conv2DConv2D!max_pooling2d_40/MaxPool:output:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2
conv2d_41/Conv2Dл
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp▒
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
conv2d_41/BiasAdd
conv2d_41/ReluReluconv2d_41/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
conv2d_41/Relu╦
max_pooling2d_41/MaxPoolMaxPoolconv2d_41/Relu:activations:0*0
_output_shapes
:         

А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_41/MaxPoolu
flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"     2  2
flatten_13/Constд
flatten_13/ReshapeReshape!max_pooling2d_41/MaxPool:output:0flatten_13/Const:output:0*
T0*(
_output_shapes
:         Аd2
flatten_13/Reshapeй
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	Аdx*
dtype02 
dense_39/MatMul/ReadVariableOpг
dense_39/MatMulMatMulflatten_13/Reshape:output:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
dense_39/MatMulз
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02!
dense_39/BiasAdd/ReadVariableOpе
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
dense_39/BiasAdds
dense_39/ReluReludense_39/BiasAdd:output:0*
T0*'
_output_shapes
:         x2
dense_39/Reluи
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:x<*
dtype02 
dense_40/MatMul/ReadVariableOpг
dense_40/MatMulMatMuldense_39/Relu:activations:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         <2
dense_40/MatMulз
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02!
dense_40/BiasAdd/ReadVariableOpе
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         <2
dense_40/BiasAdds
dense_40/ReluReludense_40/BiasAdd:output:0*
T0*'
_output_shapes
:         <2
dense_40/Reluy
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_13/dropout/Constй
dropout_13/dropout/MulMuldense_40/Relu:activations:0!dropout_13/dropout/Const:output:0*
T0*'
_output_shapes
:         <2
dropout_13/dropout/Mul
dropout_13/dropout/ShapeShapedense_40/Relu:activations:0*
T0*
_output_shapes
:2
dropout_13/dropout/Shape╒
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*'
_output_shapes
:         <*
dtype021
/dropout_13/dropout/random_uniform/RandomUniformЛ
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_13/dropout/GreaterEqual/yъ
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         <2!
dropout_13/dropout/GreaterEqualа
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         <2
dropout_13/dropout/Castж
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*'
_output_shapes
:         <2
dropout_13/dropout/Mul_1и
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02 
dense_41/MatMul/ReadVariableOpд
dense_41/MatMulMatMuldropout_13/dropout/Mul_1:z:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_41/MatMulз
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_41/BiasAdd/ReadVariableOpе
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_41/BiasAdd|
dense_41/SoftmaxSoftmaxdense_41/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_41/SoftmaxЖ
IdentityIdentitydense_41/Softmax:softmax:0!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         АА::::::::::::2D
 conv2d_39/BiasAdd/ReadVariableOp conv2d_39/BiasAdd/ReadVariableOp2B
conv2d_39/Conv2D/ReadVariableOpconv2d_39/Conv2D/ReadVariableOp2D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
А
c
D__inference_dropout_13_layer_call_and_return_conditional_losses_5495

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         <2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         <*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         <2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         <2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         <2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         <2

Identity"
identityIdentity:output:0*&
_input_shapes
:         <:O K
'
_output_shapes
:         <
 
_user_specified_nameinputs
╝	
Ш
,__inference_sequential_13_layer_call_fn_5925

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_13_layer_call_and_return_conditional_losses_56902
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         АА::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
╙

▄
C__inference_conv2d_41_layer_call_and_return_conditional_losses_5976

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:           А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:           @
 
_user_specified_nameinputs
╫	
б
,__inference_sequential_13_layer_call_fn_5649
conv2d_39_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallconv2d_39_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_13_layer_call_and_return_conditional_losses_56222
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         АА::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:         АА
)
_user_specified_nameconv2d_39_input
А
c
D__inference_dropout_13_layer_call_and_return_conditional_losses_6048

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         <2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         <*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         <2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         <2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         <2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         <2

Identity"
identityIdentity:output:0*&
_input_shapes
:         <:O K
'
_output_shapes
:         <
 
_user_specified_nameinputs
Е
}
(__inference_conv2d_39_layer_call_fn_5945

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_39_layer_call_and_return_conditional_losses_53422
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:         АА 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         АА::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
о
K
/__inference_max_pooling2d_39_layer_call_fn_5303

inputs
identityю
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_52972
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╛
`
D__inference_flatten_13_layer_call_and_return_conditional_losses_5421

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     2  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         Аd2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         Аd2

Identity"
identityIdentity:output:0*/
_input_shapes
:         

А:X T
0
_output_shapes
:         

А
 
_user_specified_nameinputs
═

▄
C__inference_conv2d_40_layer_call_and_return_conditional_losses_5956

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @@@2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @@ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @@ 
 
_user_specified_nameinputs
я1
╙
G__inference_sequential_13_layer_call_and_return_conditional_losses_5541
conv2d_39_input
conv2d_39_5353
conv2d_39_5355
conv2d_40_5381
conv2d_40_5383
conv2d_41_5409
conv2d_41_5411
dense_39_5451
dense_39_5453
dense_40_5478
dense_40_5480
dense_41_5535
dense_41_5537
identityИв!conv2d_39/StatefulPartitionedCallв!conv2d_40/StatefulPartitionedCallв!conv2d_41/StatefulPartitionedCallв dense_39/StatefulPartitionedCallв dense_40/StatefulPartitionedCallв dense_41/StatefulPartitionedCallв"dropout_13/StatefulPartitionedCallй
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCallconv2d_39_inputconv2d_39_5353conv2d_39_5355*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_39_layer_call_and_return_conditional_losses_53422#
!conv2d_39/StatefulPartitionedCallЩ
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_52972"
 max_pooling2d_39/PartitionedCall┴
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0conv2d_40_5381conv2d_40_5383*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_40_layer_call_and_return_conditional_losses_53702#
!conv2d_40/StatefulPartitionedCallЩ
 max_pooling2d_40/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_53092"
 max_pooling2d_40/PartitionedCall┬
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0conv2d_41_5409conv2d_41_5411*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_41_layer_call_and_return_conditional_losses_53982#
!conv2d_41/StatefulPartitionedCallЪ
 max_pooling2d_41/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         

А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_53212"
 max_pooling2d_41/PartitionedCall 
flatten_13/PartitionedCallPartitionedCall)max_pooling2d_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Аd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_13_layer_call_and_return_conditional_losses_54212
flatten_13/PartitionedCallо
 dense_39/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_39_5451dense_39_5453*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_39_layer_call_and_return_conditional_losses_54402"
 dense_39/StatefulPartitionedCall┤
 dense_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0dense_40_5478dense_40_5480*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_40_layer_call_and_return_conditional_losses_54672"
 dense_40/StatefulPartitionedCallЦ
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         <* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_13_layer_call_and_return_conditional_losses_54952$
"dropout_13/StatefulPartitionedCall╢
 dense_41/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_41_5535dense_41_5537*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_41_layer_call_and_return_conditional_losses_55242"
 dense_41/StatefulPartitionedCallў
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0"^conv2d_39/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         АА::::::::::::2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall:b ^
1
_output_shapes
:         АА
)
_user_specified_nameconv2d_39_input
╠╜
д
 __inference__traced_restore_6386
file_prefix%
!assignvariableop_conv2d_39_kernel%
!assignvariableop_1_conv2d_39_bias'
#assignvariableop_2_conv2d_40_kernel%
!assignvariableop_3_conv2d_40_bias'
#assignvariableop_4_conv2d_41_kernel%
!assignvariableop_5_conv2d_41_bias&
"assignvariableop_6_dense_39_kernel$
 assignvariableop_7_dense_39_bias&
"assignvariableop_8_dense_40_kernel$
 assignvariableop_9_dense_40_bias'
#assignvariableop_10_dense_41_kernel%
!assignvariableop_11_dense_41_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_1/
+assignvariableop_21_adam_conv2d_39_kernel_m-
)assignvariableop_22_adam_conv2d_39_bias_m/
+assignvariableop_23_adam_conv2d_40_kernel_m-
)assignvariableop_24_adam_conv2d_40_bias_m/
+assignvariableop_25_adam_conv2d_41_kernel_m-
)assignvariableop_26_adam_conv2d_41_bias_m.
*assignvariableop_27_adam_dense_39_kernel_m,
(assignvariableop_28_adam_dense_39_bias_m.
*assignvariableop_29_adam_dense_40_kernel_m,
(assignvariableop_30_adam_dense_40_bias_m.
*assignvariableop_31_adam_dense_41_kernel_m,
(assignvariableop_32_adam_dense_41_bias_m/
+assignvariableop_33_adam_conv2d_39_kernel_v-
)assignvariableop_34_adam_conv2d_39_bias_v/
+assignvariableop_35_adam_conv2d_40_kernel_v-
)assignvariableop_36_adam_conv2d_40_bias_v/
+assignvariableop_37_adam_conv2d_41_kernel_v-
)assignvariableop_38_adam_conv2d_41_bias_v.
*assignvariableop_39_adam_dense_39_kernel_v,
(assignvariableop_40_adam_dense_39_bias_v.
*assignvariableop_41_adam_dense_40_kernel_v,
(assignvariableop_42_adam_dense_40_bias_v.
*assignvariableop_43_adam_dense_41_kernel_v,
(assignvariableop_44_adam_dense_41_bias_v
identity_46ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9└
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*╠
value┬B┐.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesъ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesФ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╬
_output_shapes╗
╕::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityа
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_39_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ж
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_39_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2и
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_40_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ж
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_40_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4и
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_41_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ж
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_41_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6з
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_39_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7е
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_39_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8з
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_40_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9е
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_40_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10л
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_41_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11й
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_41_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12е
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13з
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14з
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ж
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16о
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17б
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18б
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19г
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20г
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21│
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_39_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22▒
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_39_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23│
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_40_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24▒
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_40_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25│
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_41_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26▒
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_41_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27▓
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_39_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28░
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_39_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29▓
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_40_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30░
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_40_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31▓
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_41_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32░
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_41_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33│
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_39_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▒
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_39_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35│
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_40_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36▒
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_40_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37│
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_41_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38▒
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_41_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39▓
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_39_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40░
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_39_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41▓
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_40_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42░
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_40_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43▓
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_41_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44░
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_41_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp╝
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45п
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*╦
_input_shapes╣
╢: :::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
А
f
J__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_5309

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
█
|
'__inference_dense_40_layer_call_fn_6036

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_40_layer_call_and_return_conditional_losses_54672
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         <2

Identity"
identityIdentity:output:0*.
_input_shapes
:         x::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         x
 
_user_specified_nameinputs
Ї	
█
B__inference_dense_41_layer_call_and_return_conditional_losses_5524

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         <::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         <
 
_user_specified_nameinputs
▄]
╝
__inference__traced_save_6241
file_prefix/
+savev2_conv2d_39_kernel_read_readvariableop-
)savev2_conv2d_39_bias_read_readvariableop/
+savev2_conv2d_40_kernel_read_readvariableop-
)savev2_conv2d_40_bias_read_readvariableop/
+savev2_conv2d_41_kernel_read_readvariableop-
)savev2_conv2d_41_bias_read_readvariableop.
*savev2_dense_39_kernel_read_readvariableop,
(savev2_dense_39_bias_read_readvariableop.
*savev2_dense_40_kernel_read_readvariableop,
(savev2_dense_40_bias_read_readvariableop.
*savev2_dense_41_kernel_read_readvariableop,
(savev2_dense_41_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_39_kernel_m_read_readvariableop4
0savev2_adam_conv2d_39_bias_m_read_readvariableop6
2savev2_adam_conv2d_40_kernel_m_read_readvariableop4
0savev2_adam_conv2d_40_bias_m_read_readvariableop6
2savev2_adam_conv2d_41_kernel_m_read_readvariableop4
0savev2_adam_conv2d_41_bias_m_read_readvariableop5
1savev2_adam_dense_39_kernel_m_read_readvariableop3
/savev2_adam_dense_39_bias_m_read_readvariableop5
1savev2_adam_dense_40_kernel_m_read_readvariableop3
/savev2_adam_dense_40_bias_m_read_readvariableop5
1savev2_adam_dense_41_kernel_m_read_readvariableop3
/savev2_adam_dense_41_bias_m_read_readvariableop6
2savev2_adam_conv2d_39_kernel_v_read_readvariableop4
0savev2_adam_conv2d_39_bias_v_read_readvariableop6
2savev2_adam_conv2d_40_kernel_v_read_readvariableop4
0savev2_adam_conv2d_40_bias_v_read_readvariableop6
2savev2_adam_conv2d_41_kernel_v_read_readvariableop4
0savev2_adam_conv2d_41_bias_v_read_readvariableop5
1savev2_adam_dense_39_kernel_v_read_readvariableop3
/savev2_adam_dense_39_bias_v_read_readvariableop5
1savev2_adam_dense_40_kernel_v_read_readvariableop3
/savev2_adam_dense_40_bias_v_read_readvariableop5
1savev2_adam_dense_41_kernel_v_read_readvariableop3
/savev2_adam_dense_41_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename║
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*╠
value┬B┐.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesф
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЕ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_39_kernel_read_readvariableop)savev2_conv2d_39_bias_read_readvariableop+savev2_conv2d_40_kernel_read_readvariableop)savev2_conv2d_40_bias_read_readvariableop+savev2_conv2d_41_kernel_read_readvariableop)savev2_conv2d_41_bias_read_readvariableop*savev2_dense_39_kernel_read_readvariableop(savev2_dense_39_bias_read_readvariableop*savev2_dense_40_kernel_read_readvariableop(savev2_dense_40_bias_read_readvariableop*savev2_dense_41_kernel_read_readvariableop(savev2_dense_41_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_39_kernel_m_read_readvariableop0savev2_adam_conv2d_39_bias_m_read_readvariableop2savev2_adam_conv2d_40_kernel_m_read_readvariableop0savev2_adam_conv2d_40_bias_m_read_readvariableop2savev2_adam_conv2d_41_kernel_m_read_readvariableop0savev2_adam_conv2d_41_bias_m_read_readvariableop1savev2_adam_dense_39_kernel_m_read_readvariableop/savev2_adam_dense_39_bias_m_read_readvariableop1savev2_adam_dense_40_kernel_m_read_readvariableop/savev2_adam_dense_40_bias_m_read_readvariableop1savev2_adam_dense_41_kernel_m_read_readvariableop/savev2_adam_dense_41_bias_m_read_readvariableop2savev2_adam_conv2d_39_kernel_v_read_readvariableop0savev2_adam_conv2d_39_bias_v_read_readvariableop2savev2_adam_conv2d_40_kernel_v_read_readvariableop0savev2_adam_conv2d_40_bias_v_read_readvariableop2savev2_adam_conv2d_41_kernel_v_read_readvariableop0savev2_adam_conv2d_41_bias_v_read_readvariableop1savev2_adam_dense_39_kernel_v_read_readvariableop/savev2_adam_dense_39_bias_v_read_readvariableop1savev2_adam_dense_40_kernel_v_read_readvariableop/savev2_adam_dense_40_bias_v_read_readvariableop1savev2_adam_dense_41_kernel_v_read_readvariableop/savev2_adam_dense_41_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ь
_input_shapesК
З: : : : @:@:@А:А:	Аdx:x:x<:<:<:: : : : : : : : : : : : @:@:@А:А:	Аdx:x:x<:<:<:: : : @:@:@А:А:	Аdx:x:x<:<:<:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:%!

_output_shapes
:	Аdx: 

_output_shapes
:x:$	 

_output_shapes

:x<: 


_output_shapes
:<:$ 

_output_shapes

:<: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:%!

_output_shapes
:	Аdx: 

_output_shapes
:x:$ 

_output_shapes

:x<: 

_output_shapes
:<:$  

_output_shapes

:<: !

_output_shapes
::,"(
&
_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
: @: %

_output_shapes
:@:-&)
'
_output_shapes
:@А:!'

_output_shapes	
:А:%(!

_output_shapes
:	Аdx: )

_output_shapes
:x:$* 

_output_shapes

:x<: +

_output_shapes
:<:$, 

_output_shapes

:<: -

_output_shapes
::.

_output_shapes
: 
й
E
)__inference_flatten_13_layer_call_fn_5996

inputs
identity╞
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Аd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_13_layer_call_and_return_conditional_losses_54212
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         Аd2

Identity"
identityIdentity:output:0*/
_input_shapes
:         

А:X T
0
_output_shapes
:         

А
 
_user_specified_nameinputs
█
|
'__inference_dense_41_layer_call_fn_6083

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_41_layer_call_and_return_conditional_losses_55242
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         <::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         <
 
_user_specified_nameinputs
╟
b
D__inference_dropout_13_layer_call_and_return_conditional_losses_5500

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         <2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         <2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         <:O K
'
_output_shapes
:         <
 
_user_specified_nameinputs
╗0
о
G__inference_sequential_13_layer_call_and_return_conditional_losses_5580
conv2d_39_input
conv2d_39_5544
conv2d_39_5546
conv2d_40_5550
conv2d_40_5552
conv2d_41_5556
conv2d_41_5558
dense_39_5563
dense_39_5565
dense_40_5568
dense_40_5570
dense_41_5574
dense_41_5576
identityИв!conv2d_39/StatefulPartitionedCallв!conv2d_40/StatefulPartitionedCallв!conv2d_41/StatefulPartitionedCallв dense_39/StatefulPartitionedCallв dense_40/StatefulPartitionedCallв dense_41/StatefulPartitionedCallй
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCallconv2d_39_inputconv2d_39_5544conv2d_39_5546*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_39_layer_call_and_return_conditional_losses_53422#
!conv2d_39/StatefulPartitionedCallЩ
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_52972"
 max_pooling2d_39/PartitionedCall┴
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0conv2d_40_5550conv2d_40_5552*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_40_layer_call_and_return_conditional_losses_53702#
!conv2d_40/StatefulPartitionedCallЩ
 max_pooling2d_40/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_53092"
 max_pooling2d_40/PartitionedCall┬
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0conv2d_41_5556conv2d_41_5558*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_41_layer_call_and_return_conditional_losses_53982#
!conv2d_41/StatefulPartitionedCallЪ
 max_pooling2d_41/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         

А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_53212"
 max_pooling2d_41/PartitionedCall 
flatten_13/PartitionedCallPartitionedCall)max_pooling2d_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Аd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_13_layer_call_and_return_conditional_losses_54212
flatten_13/PartitionedCallо
 dense_39/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_39_5563dense_39_5565*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_39_layer_call_and_return_conditional_losses_54402"
 dense_39/StatefulPartitionedCall┤
 dense_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0dense_40_5568dense_40_5570*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_40_layer_call_and_return_conditional_losses_54672"
 dense_40/StatefulPartitionedCall■
dropout_13/PartitionedCallPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         <* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_13_layer_call_and_return_conditional_losses_55002
dropout_13/PartitionedCallо
 dense_41/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_41_5574dense_41_5576*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_41_layer_call_and_return_conditional_losses_55242"
 dense_41/StatefulPartitionedCall╥
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0"^conv2d_39/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         АА::::::::::::2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall:b ^
1
_output_shapes
:         АА
)
_user_specified_nameconv2d_39_input
е	
Ч
"__inference_signature_wrapper_5756
conv2d_39_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallconv2d_39_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__wrapped_model_52912
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         АА::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:         АА
)
_user_specified_nameconv2d_39_input
я	
█
B__inference_dense_39_layer_call_and_return_conditional_losses_5440

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Аdx*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         x2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         x2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Аd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Аd
 
_user_specified_nameinputs
¤
}
(__inference_conv2d_40_layer_call_fn_5965

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_40_layer_call_and_return_conditional_losses_53702
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @@ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @@ 
 
_user_specified_nameinputs
╝	
Ш
,__inference_sequential_13_layer_call_fn_5896

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_13_layer_call_and_return_conditional_losses_56222
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         АА::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
╘1
╩
G__inference_sequential_13_layer_call_and_return_conditional_losses_5622

inputs
conv2d_39_5586
conv2d_39_5588
conv2d_40_5592
conv2d_40_5594
conv2d_41_5598
conv2d_41_5600
dense_39_5605
dense_39_5607
dense_40_5610
dense_40_5612
dense_41_5616
dense_41_5618
identityИв!conv2d_39/StatefulPartitionedCallв!conv2d_40/StatefulPartitionedCallв!conv2d_41/StatefulPartitionedCallв dense_39/StatefulPartitionedCallв dense_40/StatefulPartitionedCallв dense_41/StatefulPartitionedCallв"dropout_13/StatefulPartitionedCallа
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_39_5586conv2d_39_5588*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_39_layer_call_and_return_conditional_losses_53422#
!conv2d_39/StatefulPartitionedCallЩ
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_52972"
 max_pooling2d_39/PartitionedCall┴
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0conv2d_40_5592conv2d_40_5594*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_40_layer_call_and_return_conditional_losses_53702#
!conv2d_40/StatefulPartitionedCallЩ
 max_pooling2d_40/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_53092"
 max_pooling2d_40/PartitionedCall┬
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0conv2d_41_5598conv2d_41_5600*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_41_layer_call_and_return_conditional_losses_53982#
!conv2d_41/StatefulPartitionedCallЪ
 max_pooling2d_41/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         

А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_53212"
 max_pooling2d_41/PartitionedCall 
flatten_13/PartitionedCallPartitionedCall)max_pooling2d_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Аd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_13_layer_call_and_return_conditional_losses_54212
flatten_13/PartitionedCallо
 dense_39/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_39_5605dense_39_5607*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_39_layer_call_and_return_conditional_losses_54402"
 dense_39/StatefulPartitionedCall┤
 dense_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0dense_40_5610dense_40_5612*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_40_layer_call_and_return_conditional_losses_54672"
 dense_40/StatefulPartitionedCallЦ
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         <* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_13_layer_call_and_return_conditional_losses_54952$
"dropout_13/StatefulPartitionedCall╢
 dense_41/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_41_5616dense_41_5618*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_41_layer_call_and_return_conditional_losses_55242"
 dense_41/StatefulPartitionedCallў
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0"^conv2d_39/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         АА::::::::::::2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
ь	
█
B__inference_dense_40_layer_call_and_return_conditional_losses_5467

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         <2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         <2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         <2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         <2

Identity"
identityIdentity:output:0*.
_input_shapes
:         x::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         x
 
_user_specified_nameinputs
╙

▄
C__inference_conv2d_41_layer_call_and_return_conditional_losses_5398

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:           А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:           @
 
_user_specified_nameinputs
А
f
J__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_5297

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
═

▄
C__inference_conv2d_40_layer_call_and_return_conditional_losses_5370

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @@@2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @@ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @@ 
 
_user_specified_nameinputs"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┼
serving_default▒
U
conv2d_39_inputB
!serving_default_conv2d_39_input:0         АА<
dense_410
StatefulPartitionedCall:0         tensorflow/serving/predict:С┌
ЮV
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
о__call__
п_default_save_signature
+░&call_and_return_all_conditional_losses"дR
_tf_keras_sequentialЕR{"class_name": "Sequential", "name": "sequential_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_13", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_39_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_40", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_13", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_13", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_39_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_40", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_13", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
√


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses"╘	
_tf_keras_layer║	{"class_name": "Conv2D", "name": "conv2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_39", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}}
Г
	variables
regularization_losses
trainable_variables
	keras_api
│__call__
+┤&call_and_return_all_conditional_losses"Є
_tf_keras_layer╪{"class_name": "MaxPooling2D", "name": "max_pooling2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ў	

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
╡__call__
+╢&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"class_name": "Conv2D", "name": "conv2d_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_40", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 32]}}
Г
"	variables
#regularization_losses
$trainable_variables
%	keras_api
╖__call__
+╕&call_and_return_all_conditional_losses"Є
_tf_keras_layer╪{"class_name": "MaxPooling2D", "name": "max_pooling2d_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ў	

&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
╣__call__
+║&call_and_return_all_conditional_losses"╨
_tf_keras_layer╢{"class_name": "Conv2D", "name": "conv2d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
Г
,	variables
-regularization_losses
.trainable_variables
/	keras_api
╗__call__
+╝&call_and_return_all_conditional_losses"Є
_tf_keras_layer╪{"class_name": "MaxPooling2D", "name": "max_pooling2d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ъ
0	variables
1regularization_losses
2trainable_variables
3	keras_api
╜__call__
+╛&call_and_return_all_conditional_losses"┘
_tf_keras_layer┐{"class_name": "Flatten", "name": "flatten_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_13", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
√

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
┐__call__
+└&call_and_return_all_conditional_losses"╘
_tf_keras_layer║{"class_name": "Dense", "name": "dense_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12800}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12800]}}
Ў

:kernel
;bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
┴__call__
+┬&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"class_name": "Dense", "name": "dense_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 120]}}
щ
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
├__call__
+─&call_and_return_all_conditional_losses"╪
_tf_keras_layer╛{"class_name": "Dropout", "name": "dropout_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
Ў

Dkernel
Ebias
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
┼__call__
+╞&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"class_name": "Dense", "name": "dense_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
├
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratemЦmЧmШmЩ&mЪ'mЫ4mЬ5mЭ:mЮ;mЯDmаEmбvвvгvдvе&vж'vз4vи5vй:vк;vлDvмEvн"
	optimizer
v
0
1
2
3
&4
'5
46
57
:8
;9
D10
E11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
&4
'5
46
57
:8
;9
D10
E11"
trackable_list_wrapper
╬
	variables

Olayers
Pmetrics
regularization_losses
trainable_variables
Qlayer_regularization_losses
Rlayer_metrics
Snon_trainable_variables
о__call__
п_default_save_signature
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
-
╟serving_default"
signature_map
*:( 2conv2d_39/kernel
: 2conv2d_39/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
	variables

Tlayers
Umetrics
regularization_losses
trainable_variables
Vlayer_regularization_losses
Wlayer_metrics
Xnon_trainable_variables
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
	variables

Ylayers
Zmetrics
regularization_losses
trainable_variables
[layer_regularization_losses
\layer_metrics
]non_trainable_variables
│__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_40/kernel
:@2conv2d_40/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
	variables

^layers
_metrics
regularization_losses
 trainable_variables
`layer_regularization_losses
alayer_metrics
bnon_trainable_variables
╡__call__
+╢&call_and_return_all_conditional_losses
'╢"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
"	variables

clayers
dmetrics
#regularization_losses
$trainable_variables
elayer_regularization_losses
flayer_metrics
gnon_trainable_variables
╖__call__
+╕&call_and_return_all_conditional_losses
'╕"call_and_return_conditional_losses"
_generic_user_object
+:)@А2conv2d_41/kernel
:А2conv2d_41/bias
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
░
(	variables

hlayers
imetrics
)regularization_losses
*trainable_variables
jlayer_regularization_losses
klayer_metrics
lnon_trainable_variables
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
,	variables

mlayers
nmetrics
-regularization_losses
.trainable_variables
olayer_regularization_losses
player_metrics
qnon_trainable_variables
╗__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
0	variables

rlayers
smetrics
1regularization_losses
2trainable_variables
tlayer_regularization_losses
ulayer_metrics
vnon_trainable_variables
╜__call__
+╛&call_and_return_all_conditional_losses
'╛"call_and_return_conditional_losses"
_generic_user_object
": 	Аdx2dense_39/kernel
:x2dense_39/bias
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
░
6	variables

wlayers
xmetrics
7regularization_losses
8trainable_variables
ylayer_regularization_losses
zlayer_metrics
{non_trainable_variables
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
!:x<2dense_40/kernel
:<2dense_40/bias
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
▒
<	variables

|layers
}metrics
=regularization_losses
>trainable_variables
~layer_regularization_losses
layer_metrics
Аnon_trainable_variables
┴__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
@	variables
Бlayers
Вmetrics
Aregularization_losses
Btrainable_variables
 Гlayer_regularization_losses
Дlayer_metrics
Еnon_trainable_variables
├__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
!:<2dense_41/kernel
:2dense_41/bias
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
╡
F	variables
Жlayers
Зmetrics
Gregularization_losses
Htrainable_variables
 Иlayer_regularization_losses
Йlayer_metrics
Кnon_trainable_variables
┼__call__
+╞&call_and_return_all_conditional_losses
'╞"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
0
Л0
М1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
┐

Нtotal

Оcount
П	variables
Р	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Д

Сtotal

Тcount
У
_fn_kwargs
Ф	variables
Х	keras_api"╕
_tf_keras_metricЭ{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
Н0
О1"
trackable_list_wrapper
.
П	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
С0
Т1"
trackable_list_wrapper
.
Ф	variables"
_generic_user_object
/:- 2Adam/conv2d_39/kernel/m
!: 2Adam/conv2d_39/bias/m
/:- @2Adam/conv2d_40/kernel/m
!:@2Adam/conv2d_40/bias/m
0:.@А2Adam/conv2d_41/kernel/m
": А2Adam/conv2d_41/bias/m
':%	Аdx2Adam/dense_39/kernel/m
 :x2Adam/dense_39/bias/m
&:$x<2Adam/dense_40/kernel/m
 :<2Adam/dense_40/bias/m
&:$<2Adam/dense_41/kernel/m
 :2Adam/dense_41/bias/m
/:- 2Adam/conv2d_39/kernel/v
!: 2Adam/conv2d_39/bias/v
/:- @2Adam/conv2d_40/kernel/v
!:@2Adam/conv2d_40/bias/v
0:.@А2Adam/conv2d_41/kernel/v
": А2Adam/conv2d_41/bias/v
':%	Аdx2Adam/dense_39/kernel/v
 :x2Adam/dense_39/bias/v
&:$x<2Adam/dense_40/kernel/v
 :<2Adam/dense_40/bias/v
&:$<2Adam/dense_41/kernel/v
 :2Adam/dense_41/bias/v
■2√
,__inference_sequential_13_layer_call_fn_5925
,__inference_sequential_13_layer_call_fn_5649
,__inference_sequential_13_layer_call_fn_5717
,__inference_sequential_13_layer_call_fn_5896└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
я2ь
__inference__wrapped_model_5291╚
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0
conv2d_39_input         АА
ъ2ч
G__inference_sequential_13_layer_call_and_return_conditional_losses_5815
G__inference_sequential_13_layer_call_and_return_conditional_losses_5541
G__inference_sequential_13_layer_call_and_return_conditional_losses_5867
G__inference_sequential_13_layer_call_and_return_conditional_losses_5580└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥2╧
(__inference_conv2d_39_layer_call_fn_5945в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_conv2d_39_layer_call_and_return_conditional_losses_5936в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ч2Ф
/__inference_max_pooling2d_39_layer_call_fn_5303р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
▓2п
J__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_5297р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╥2╧
(__inference_conv2d_40_layer_call_fn_5965в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_conv2d_40_layer_call_and_return_conditional_losses_5956в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ч2Ф
/__inference_max_pooling2d_40_layer_call_fn_5315р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
▓2п
J__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_5309р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╥2╧
(__inference_conv2d_41_layer_call_fn_5985в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_conv2d_41_layer_call_and_return_conditional_losses_5976в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ч2Ф
/__inference_max_pooling2d_41_layer_call_fn_5327р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
▓2п
J__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_5321р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╙2╨
)__inference_flatten_13_layer_call_fn_5996в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_flatten_13_layer_call_and_return_conditional_losses_5991в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_dense_39_layer_call_fn_6016в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_dense_39_layer_call_and_return_conditional_losses_6007в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_dense_40_layer_call_fn_6036в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_dense_40_layer_call_and_return_conditional_losses_6027в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Р2Н
)__inference_dropout_13_layer_call_fn_6063
)__inference_dropout_13_layer_call_fn_6058┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╞2├
D__inference_dropout_13_layer_call_and_return_conditional_losses_6053
D__inference_dropout_13_layer_call_and_return_conditional_losses_6048┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╤2╬
'__inference_dense_41_layer_call_fn_6083в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_dense_41_layer_call_and_return_conditional_losses_6074в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤B╬
"__inference_signature_wrapper_5756conv2d_39_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 л
__inference__wrapped_model_5291З&'45:;DEBв?
8в5
3К0
conv2d_39_input         АА
к "3к0
.
dense_41"К
dense_41         ╖
C__inference_conv2d_39_layer_call_and_return_conditional_losses_5936p9в6
/в,
*К'
inputs         АА
к "/в,
%К"
0         АА 
Ъ П
(__inference_conv2d_39_layer_call_fn_5945c9в6
/в,
*К'
inputs         АА
к ""К         АА │
C__inference_conv2d_40_layer_call_and_return_conditional_losses_5956l7в4
-в*
(К%
inputs         @@ 
к "-в*
#К 
0         @@@
Ъ Л
(__inference_conv2d_40_layer_call_fn_5965_7в4
-в*
(К%
inputs         @@ 
к " К         @@@┤
C__inference_conv2d_41_layer_call_and_return_conditional_losses_5976m&'7в4
-в*
(К%
inputs           @
к ".в+
$К!
0           А
Ъ М
(__inference_conv2d_41_layer_call_fn_5985`&'7в4
-в*
(К%
inputs           @
к "!К           Аг
B__inference_dense_39_layer_call_and_return_conditional_losses_6007]450в-
&в#
!К
inputs         Аd
к "%в"
К
0         x
Ъ {
'__inference_dense_39_layer_call_fn_6016P450в-
&в#
!К
inputs         Аd
к "К         xв
B__inference_dense_40_layer_call_and_return_conditional_losses_6027\:;/в,
%в"
 К
inputs         x
к "%в"
К
0         <
Ъ z
'__inference_dense_40_layer_call_fn_6036O:;/в,
%в"
 К
inputs         x
к "К         <в
B__inference_dense_41_layer_call_and_return_conditional_losses_6074\DE/в,
%в"
 К
inputs         <
к "%в"
К
0         
Ъ z
'__inference_dense_41_layer_call_fn_6083ODE/в,
%в"
 К
inputs         <
к "К         д
D__inference_dropout_13_layer_call_and_return_conditional_losses_6048\3в0
)в&
 К
inputs         <
p
к "%в"
К
0         <
Ъ д
D__inference_dropout_13_layer_call_and_return_conditional_losses_6053\3в0
)в&
 К
inputs         <
p 
к "%в"
К
0         <
Ъ |
)__inference_dropout_13_layer_call_fn_6058O3в0
)в&
 К
inputs         <
p
к "К         <|
)__inference_dropout_13_layer_call_fn_6063O3в0
)в&
 К
inputs         <
p 
к "К         <к
D__inference_flatten_13_layer_call_and_return_conditional_losses_5991b8в5
.в+
)К&
inputs         

А
к "&в#
К
0         Аd
Ъ В
)__inference_flatten_13_layer_call_fn_5996U8в5
.в+
)К&
inputs         

А
к "К         Аdэ
J__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_5297ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┼
/__inference_max_pooling2d_39_layer_call_fn_5303СRвO
HвE
CК@
inputs4                                    
к ";К84                                    э
J__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_5309ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┼
/__inference_max_pooling2d_40_layer_call_fn_5315СRвO
HвE
CК@
inputs4                                    
к ";К84                                    э
J__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_5321ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┼
/__inference_max_pooling2d_41_layer_call_fn_5327СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ═
G__inference_sequential_13_layer_call_and_return_conditional_losses_5541Б&'45:;DEJвG
@в=
3К0
conv2d_39_input         АА
p

 
к "%в"
К
0         
Ъ ═
G__inference_sequential_13_layer_call_and_return_conditional_losses_5580Б&'45:;DEJвG
@в=
3К0
conv2d_39_input         АА
p 

 
к "%в"
К
0         
Ъ ├
G__inference_sequential_13_layer_call_and_return_conditional_losses_5815x&'45:;DEAв>
7в4
*К'
inputs         АА
p

 
к "%в"
К
0         
Ъ ├
G__inference_sequential_13_layer_call_and_return_conditional_losses_5867x&'45:;DEAв>
7в4
*К'
inputs         АА
p 

 
к "%в"
К
0         
Ъ д
,__inference_sequential_13_layer_call_fn_5649t&'45:;DEJвG
@в=
3К0
conv2d_39_input         АА
p

 
к "К         д
,__inference_sequential_13_layer_call_fn_5717t&'45:;DEJвG
@в=
3К0
conv2d_39_input         АА
p 

 
к "К         Ы
,__inference_sequential_13_layer_call_fn_5896k&'45:;DEAв>
7в4
*К'
inputs         АА
p

 
к "К         Ы
,__inference_sequential_13_layer_call_fn_5925k&'45:;DEAв>
7в4
*К'
inputs         АА
p 

 
к "К         ┴
"__inference_signature_wrapper_5756Ъ&'45:;DEUвR
в 
KкH
F
conv2d_39_input3К0
conv2d_39_input         АА"3к0
.
dense_41"К
dense_41         