Го
»Ђ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
0
Sigmoid
x"T
y"T"
Ttype:

2
Ѕ
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
executor_typestring И®
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
 И"serve*2.10.02unknown8З”
Д
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/output/kernel/v
}
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes

:*
dtype0
А
Adam/hidden_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/hidden_1/bias/v
y
(Adam/hidden_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_1/bias/v*
_output_shapes
:*
dtype0
И
Adam/hidden_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/hidden_1/kernel/v
Б
*Adam/hidden_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_1/kernel/v*
_output_shapes

:
*
dtype0
А
Adam/hidden_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/hidden_0/bias/v
y
(Adam/hidden_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_0/bias/v*
_output_shapes
:
*
dtype0
Й
Adam/hidden_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	†
*'
shared_nameAdam/hidden_0/kernel/v
В
*Adam/hidden_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_0/kernel/v*
_output_shapes
:	†
*
dtype0
Д
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/output/kernel/m
}
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes

:*
dtype0
А
Adam/hidden_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/hidden_1/bias/m
y
(Adam/hidden_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_1/bias/m*
_output_shapes
:*
dtype0
И
Adam/hidden_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/hidden_1/kernel/m
Б
*Adam/hidden_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_1/kernel/m*
_output_shapes

:
*
dtype0
А
Adam/hidden_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/hidden_0/bias/m
y
(Adam/hidden_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_0/bias/m*
_output_shapes
:
*
dtype0
Й
Adam/hidden_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	†
*'
shared_nameAdam/hidden_0/kernel/m
В
*Adam/hidden_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_0/kernel/m*
_output_shapes
:	†
*
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
|
sum_squared_errorsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namesum_squared_errors
u
&sum_squared_errors/Read/ReadVariableOpReadVariableOpsum_squared_errors*
_output_shapes
:*
dtype0
^
sumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namesum
W
sum/Read/ReadVariableOpReadVariableOpsum*
_output_shapes
:*
dtype0
n
sum_squaresVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namesum_squares
g
sum_squares/Read/ReadVariableOpReadVariableOpsum_squares*
_output_shapes
:*
dtype0
Z
NVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameN
S
N/Read/ReadVariableOpReadVariableOpN*
_output_shapes
:*
dtype0
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
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:*
dtype0
r
hidden_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehidden_1/bias
k
!hidden_1/bias/Read/ReadVariableOpReadVariableOphidden_1/bias*
_output_shapes
:*
dtype0
z
hidden_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namehidden_1/kernel
s
#hidden_1/kernel/Read/ReadVariableOpReadVariableOphidden_1/kernel*
_output_shapes

:
*
dtype0
r
hidden_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namehidden_0/bias
k
!hidden_0/bias/Read/ReadVariableOpReadVariableOphidden_0/bias*
_output_shapes
:
*
dtype0
{
hidden_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	†
* 
shared_namehidden_0/kernel
t
#hidden_0/kernel/Read/ReadVariableOpReadVariableOphidden_0/kernel*
_output_shapes
:	†
*
dtype0
|
serving_default_input_1Placeholder*(
_output_shapes
:€€€€€€€€€†*
dtype0*
shape:€€€€€€€€€†
К
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1hidden_0/kernelhidden_0/biashidden_1/kernelhidden_1/biasoutput/kernel*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_48359

NoOpNoOp
‘/
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*П/
valueЕ/BВ/ Bы.
Ѕ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*
¶
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¶
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
Ь
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel*
'
0
1
2
3
#4*
'
0
1
2
3
#4*
* 
∞
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
)trace_0
*trace_1
+trace_2
,trace_3* 
6
-trace_0
.trace_1
/trace_2
0trace_3* 
* 
Ю
1iter

2beta_1

3beta_2
	4decay
5learning_ratem]m^m_m`#mavbvcvdve#vf*

6serving_default* 

0
1*

0
1*
* 
У
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

<trace_0* 

=trace_0* 
_Y
VARIABLE_VALUEhidden_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEhidden_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
У
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 
_Y
VARIABLE_VALUEhidden_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEhidden_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

#0*

#0*
* 
У
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

Jtrace_0* 

Ktrace_0* 
]W
VARIABLE_VALUEoutput/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

L0
M1
N2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
O	variables
P	keras_api
	Qtotal
	Rcount*
[
S	variables
T	keras_api
UN
Vsum_squares
Wsum
Xsum_squared_errors*
8
Y	variables
Z	keras_api
	[total
	\count*

Q0
R1*

O	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
 
U0
V1
W2
X3*

S	variables*
KE
VARIABLE_VALUEN0keras_api/metrics/1/N/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEsum_squares:keras_api/metrics/1/sum_squares/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEsum2keras_api/metrics/1/sum/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEsum_squared_errorsAkeras_api/metrics/1/sum_squared_errors/.ATTRIBUTES/VARIABLE_VALUE*

[0
\1*

Y	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/hidden_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/hidden_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/hidden_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/hidden_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/hidden_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/hidden_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/hidden_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/hidden_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
І

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#hidden_0/kernel/Read/ReadVariableOp!hidden_0/bias/Read/ReadVariableOp#hidden_1/kernel/Read/ReadVariableOp!hidden_1/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpN/Read/ReadVariableOpsum_squares/Read/ReadVariableOpsum/Read/ReadVariableOp&sum_squared_errors/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/hidden_0/kernel/m/Read/ReadVariableOp(Adam/hidden_0/bias/m/Read/ReadVariableOp*Adam/hidden_1/kernel/m/Read/ReadVariableOp(Adam/hidden_1/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp*Adam/hidden_0/kernel/v/Read/ReadVariableOp(Adam/hidden_0/bias/v/Read/ReadVariableOp*Adam/hidden_1/kernel/v/Read/ReadVariableOp(Adam/hidden_1/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOpConst*)
Tin"
 2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_48814
т
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_0/kernelhidden_0/biashidden_1/kernelhidden_1/biasoutput/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1Nsum_squaressumsum_squared_errorstotalcountAdam/hidden_0/kernel/mAdam/hidden_0/bias/mAdam/hidden_1/kernel/mAdam/hidden_1/bias/mAdam/output/kernel/mAdam/hidden_0/kernel/vAdam/hidden_0/bias/vAdam/hidden_1/kernel/vAdam/hidden_1/bias/vAdam/output/kernel/v*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_48908шџ
Ђ
ч
C__inference_hidden_0_layer_call_and_return_conditional_losses_48157

inputs1
matmul_readvariableop_resource:	†
-
biasadd_readvariableop_resource:


identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	†
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€
]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€
Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€
©
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-48149*:
_output_shapes(
&:€€€€€€€€€
:€€€€€€€€€
c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€†: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
“
Л
"__inference_internal_grad_fn_48712
result_grads_0
result_grads_1
mul_hidden_0_beta
mul_hidden_0_biasadd
identityv
mulMulmul_hidden_0_betamul_hidden_0_biasadd^result_grads_0*
T0*'
_output_shapes
:€€€€€€€€€
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€
g
mul_1Mulmul_hidden_0_betamul_hidden_0_biasadd*
T0*'
_output_shapes
:€€€€€€€€€
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:€€€€€€€€€
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€
"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€
:€€€€€€€€€
: :€€€€€€€€€
:W S
'
_output_shapes
:€€€€€€€€€

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:€€€€€€€€€

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€

“
Л
"__inference_internal_grad_fn_48730
result_grads_0
result_grads_1
mul_hidden_1_beta
mul_hidden_1_biasadd
identityv
mulMulmul_hidden_1_betamul_hidden_1_biasadd^result_grads_0*
T0*'
_output_shapes
:€€€€€€€€€M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€g
mul_1Mulmul_hidden_1_betamul_hidden_1_biasadd*
T0*'
_output_shapes
:€€€€€€€€€J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€:€€€€€€€€€: :€€€€€€€€€:W S
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€
“
Л
"__inference_internal_grad_fn_48676
result_grads_0
result_grads_1
mul_hidden_0_beta
mul_hidden_0_biasadd
identityv
mulMulmul_hidden_0_betamul_hidden_0_biasadd^result_grads_0*
T0*'
_output_shapes
:€€€€€€€€€
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€
g
mul_1Mulmul_hidden_0_betamul_hidden_0_biasadd*
T0*'
_output_shapes
:€€€€€€€€€
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:€€€€€€€€€
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€
"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€
:€€€€€€€€€
: :€€€€€€€€€
:W S
'
_output_shapes
:€€€€€€€€€

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:€€€€€€€€€

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€

Я<
Р
__inference__traced_save_48814
file_prefix.
*savev2_hidden_0_kernel_read_readvariableop,
(savev2_hidden_0_bias_read_readvariableop.
*savev2_hidden_1_kernel_read_readvariableop,
(savev2_hidden_1_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop 
savev2_n_read_readvariableop*
&savev2_sum_squares_read_readvariableop"
savev2_sum_read_readvariableop1
-savev2_sum_squared_errors_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_hidden_0_kernel_m_read_readvariableop3
/savev2_adam_hidden_0_bias_m_read_readvariableop5
1savev2_adam_hidden_1_kernel_m_read_readvariableop3
/savev2_adam_hidden_1_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop5
1savev2_adam_hidden_0_kernel_v_read_readvariableop3
/savev2_adam_hidden_0_bias_v_read_readvariableop5
1savev2_adam_hidden_1_kernel_v_read_readvariableop3
/savev2_adam_hidden_1_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: А
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*©
valueЯBЬB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB0keras_api/metrics/1/N/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/1/sum_squares/.ATTRIBUTES/VARIABLE_VALUEB2keras_api/metrics/1/sum/.ATTRIBUTES/VARIABLE_VALUEBAkeras_api/metrics/1/sum_squared_errors/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHІ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B Б
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_hidden_0_kernel_read_readvariableop(savev2_hidden_0_bias_read_readvariableop*savev2_hidden_1_kernel_read_readvariableop(savev2_hidden_1_bias_read_readvariableop(savev2_output_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_n_read_readvariableop&savev2_sum_squares_read_readvariableopsavev2_sum_read_readvariableop-savev2_sum_squared_errors_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_hidden_0_kernel_m_read_readvariableop/savev2_adam_hidden_0_bias_m_read_readvariableop1savev2_adam_hidden_1_kernel_m_read_readvariableop/savev2_adam_hidden_1_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop1savev2_adam_hidden_0_kernel_v_read_readvariableop/savev2_adam_hidden_0_bias_v_read_readvariableop1savev2_adam_hidden_1_kernel_v_read_readvariableop/savev2_adam_hidden_1_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ƒ
_input_shapes≤
ѓ: :	†
:
:
::: : : : : : : ::::: : :	†
:
:
:::	†
:
:
::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	†
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	†
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

::%!

_output_shapes
:	†
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

::

_output_shapes
: 
Ђ
ч
C__inference_hidden_0_layer_call_and_return_conditional_losses_48486

inputs1
matmul_readvariableop_resource:	†
-
biasadd_readvariableop_resource:


identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	†
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€
]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€
Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€
©
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-48478*:
_output_shapes(
&:€€€€€€€€€
:€€€€€€€€€
c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€†: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
Ы
y
"__inference_internal_grad_fn_48658
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:€€€€€€€€€
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€
U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:€€€€€€€€€
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:€€€€€€€€€
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€
"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€
:€€€€€€€€€
: :€€€€€€€€€
:W S
'
_output_shapes
:€€€€€€€€€

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:€€€€€€€€€

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€

І
ц
C__inference_hidden_1_layer_call_and_return_conditional_losses_48181

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€©
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-48173*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€

 
_user_specified_nameinputs
Ы
y
"__inference_internal_grad_fn_48622
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:€€€€€€€€€M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:€€€€€€€€€J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€:€€€€€€€€€: :€€€€€€€€€:W S
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€
¶o
Ф
!__inference__traced_restore_48908
file_prefix3
 assignvariableop_hidden_0_kernel:	†
.
 assignvariableop_1_hidden_0_bias:
4
"assignvariableop_2_hidden_1_kernel:
.
 assignvariableop_3_hidden_1_bias:2
 assignvariableop_4_output_kernel:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: #
assignvariableop_12_n:-
assignvariableop_13_sum_squares:%
assignvariableop_14_sum:4
&assignvariableop_15_sum_squared_errors:#
assignvariableop_16_total: #
assignvariableop_17_count: =
*assignvariableop_18_adam_hidden_0_kernel_m:	†
6
(assignvariableop_19_adam_hidden_0_bias_m:
<
*assignvariableop_20_adam_hidden_1_kernel_m:
6
(assignvariableop_21_adam_hidden_1_bias_m::
(assignvariableop_22_adam_output_kernel_m:=
*assignvariableop_23_adam_hidden_0_kernel_v:	†
6
(assignvariableop_24_adam_hidden_0_bias_v:
<
*assignvariableop_25_adam_hidden_1_kernel_v:
6
(assignvariableop_26_adam_hidden_1_bias_v::
(assignvariableop_27_adam_output_kernel_v:
identity_29ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Г
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*©
valueЯBЬB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB0keras_api/metrics/1/N/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/1/sum_squares/.ATTRIBUTES/VARIABLE_VALUEB2keras_api/metrics/1/sum/.ATTRIBUTES/VARIABLE_VALUEBAkeras_api/metrics/1/sum_squared_errors/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH™
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B ∞
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*И
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOpAssignVariableOp assignvariableop_hidden_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_1AssignVariableOp assignvariableop_1_hidden_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_2AssignVariableOp"assignvariableop_2_hidden_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_3AssignVariableOp assignvariableop_3_hidden_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_4AssignVariableOp assignvariableop_4_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_12AssignVariableOpassignvariableop_12_nIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_13AssignVariableOpassignvariableop_13_sum_squaresIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_14AssignVariableOpassignvariableop_14_sumIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_15AssignVariableOp&assignvariableop_15_sum_squared_errorsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_hidden_0_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_hidden_0_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_hidden_1_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_hidden_1_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_output_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_hidden_0_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_hidden_0_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_hidden_1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_hidden_1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_output_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ј
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: §
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
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
 
м
*__inference_sequential_layer_call_fn_48212
input_1
unknown:	†

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_48199o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€†: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€†
!
_user_specified_name	input_1
о'
у
 __inference__wrapped_model_48132
input_1E
2sequential_hidden_0_matmul_readvariableop_resource:	†
A
3sequential_hidden_0_biasadd_readvariableop_resource:
D
2sequential_hidden_1_matmul_readvariableop_resource:
A
3sequential_hidden_1_biasadd_readvariableop_resource:B
0sequential_output_matmul_readvariableop_resource:
identityИҐ*sequential/hidden_0/BiasAdd/ReadVariableOpҐ)sequential/hidden_0/MatMul/ReadVariableOpҐ*sequential/hidden_1/BiasAdd/ReadVariableOpҐ)sequential/hidden_1/MatMul/ReadVariableOpҐ'sequential/output/MatMul/ReadVariableOpЭ
)sequential/hidden_0/MatMul/ReadVariableOpReadVariableOp2sequential_hidden_0_matmul_readvariableop_resource*
_output_shapes
:	†
*
dtype0Т
sequential/hidden_0/MatMulMatMulinput_11sequential/hidden_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
Ъ
*sequential/hidden_0/BiasAdd/ReadVariableOpReadVariableOp3sequential_hidden_0_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0≤
sequential/hidden_0/BiasAddBiasAdd$sequential/hidden_0/MatMul:product:02sequential/hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
]
sequential/hidden_0/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
sequential/hidden_0/mulMul!sequential/hidden_0/beta:output:0$sequential/hidden_0/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
u
sequential/hidden_0/SigmoidSigmoidsequential/hidden_0/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€
Щ
sequential/hidden_0/mul_1Mul$sequential/hidden_0/BiasAdd:output:0sequential/hidden_0/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€
y
sequential/hidden_0/IdentityIdentitysequential/hidden_0/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€
е
sequential/hidden_0/IdentityN	IdentityNsequential/hidden_0/mul_1:z:0$sequential/hidden_0/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-48107*:
_output_shapes(
&:€€€€€€€€€
:€€€€€€€€€
Ь
)sequential/hidden_1/MatMul/ReadVariableOpReadVariableOp2sequential_hidden_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0±
sequential/hidden_1/MatMulMatMul&sequential/hidden_0/IdentityN:output:01sequential/hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
*sequential/hidden_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_hidden_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0≤
sequential/hidden_1/BiasAddBiasAdd$sequential/hidden_1/MatMul:product:02sequential/hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€]
sequential/hidden_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
sequential/hidden_1/mulMul!sequential/hidden_1/beta:output:0$sequential/hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€u
sequential/hidden_1/SigmoidSigmoidsequential/hidden_1/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€Щ
sequential/hidden_1/mul_1Mul$sequential/hidden_1/BiasAdd:output:0sequential/hidden_1/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€y
sequential/hidden_1/IdentityIdentitysequential/hidden_1/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€е
sequential/hidden_1/IdentityN	IdentityNsequential/hidden_1/mul_1:z:0$sequential/hidden_1/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-48121*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€Ш
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0≠
sequential/output/MatMulMatMul&sequential/hidden_1/IdentityN:output:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€q
IdentityIdentity"sequential/output/MatMul:product:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ґ
NoOpNoOp+^sequential/hidden_0/BiasAdd/ReadVariableOp*^sequential/hidden_0/MatMul/ReadVariableOp+^sequential/hidden_1/BiasAdd/ReadVariableOp*^sequential/hidden_1/MatMul/ReadVariableOp(^sequential/output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€†: : : : : 2X
*sequential/hidden_0/BiasAdd/ReadVariableOp*sequential/hidden_0/BiasAdd/ReadVariableOp2V
)sequential/hidden_0/MatMul/ReadVariableOp)sequential/hidden_0/MatMul/ReadVariableOp2X
*sequential/hidden_1/BiasAdd/ReadVariableOp*sequential/hidden_1/BiasAdd/ReadVariableOp2V
)sequential/hidden_1/MatMul/ReadVariableOp)sequential/hidden_1/MatMul/ReadVariableOp2R
'sequential/output/MatMul/ReadVariableOp'sequential/output/MatMul/ReadVariableOp:Q M
(
_output_shapes
:€€€€€€€€€†
!
_user_specified_name	input_1
«
л
*__inference_sequential_layer_call_fn_48374

inputs
unknown:	†

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_48199o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€†: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
Ю
е
#__inference_signature_wrapper_48359
input_1
unknown:	†

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_48132o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€†: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€†
!
_user_specified_name	input_1
І
ц
C__inference_hidden_1_layer_call_and_return_conditional_losses_48513

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€©
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-48505*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€

 
_user_specified_nameinputs
«
л
*__inference_sequential_layer_call_fn_48389

inputs
unknown:	†

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_48274o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€†: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
±
м
E__inference_sequential_layer_call_and_return_conditional_losses_48274

inputs!
hidden_0_48260:	†

hidden_0_48262:
 
hidden_1_48265:

hidden_1_48267:
output_48270:
identityИҐ hidden_0/StatefulPartitionedCallҐ hidden_1/StatefulPartitionedCallҐoutput/StatefulPartitionedCallн
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_0_48260hidden_0_48262*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_0_layer_call_and_return_conditional_losses_48157Р
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_48265hidden_1_48267*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_1_layer_call_and_return_conditional_losses_48181ш
output/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0output_48270*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_48194v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≠
NoOpNoOp!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€†: : : : : 2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
“
Л
"__inference_internal_grad_fn_48694
result_grads_0
result_grads_1
mul_hidden_1_beta
mul_hidden_1_biasadd
identityv
mulMulmul_hidden_1_betamul_hidden_1_biasadd^result_grads_0*
T0*'
_output_shapes
:€€€€€€€€€M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€g
mul_1Mulmul_hidden_1_betamul_hidden_1_biasadd*
T0*'
_output_shapes
:€€€€€€€€€J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€:€€€€€€€€€: :€€€€€€€€€:W S
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€
 
м
*__inference_sequential_layer_call_fn_48302
input_1
unknown:	†

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_48274o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€†: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€†
!
_user_specified_name	input_1
√
Ц
(__inference_hidden_0_layer_call_fn_48468

inputs
unknown:	†

	unknown_0:

identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_0_layer_call_and_return_conditional_losses_48157o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€†: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
Х
°
"__inference_internal_grad_fn_48748
result_grads_0
result_grads_1 
mul_sequential_hidden_0_beta#
mul_sequential_hidden_0_biasadd
identityМ
mulMulmul_sequential_hidden_0_betamul_sequential_hidden_0_biasadd^result_grads_0*
T0*'
_output_shapes
:€€€€€€€€€
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€
}
mul_1Mulmul_sequential_hidden_0_betamul_sequential_hidden_0_biasadd*
T0*'
_output_shapes
:€€€€€€€€€
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:€€€€€€€€€
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€
"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€
:€€€€€€€€€
: :€€€€€€€€€
:W S
'
_output_shapes
:€€€€€€€€€

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:€€€€€€€€€

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€

Ы
y
"__inference_internal_grad_fn_48604
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:€€€€€€€€€M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:€€€€€€€€€J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€:€€€€€€€€€: :€€€€€€€€€:W S
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€
±!
©
E__inference_sequential_layer_call_and_return_conditional_losses_48424

inputs:
'hidden_0_matmul_readvariableop_resource:	†
6
(hidden_0_biasadd_readvariableop_resource:
9
'hidden_1_matmul_readvariableop_resource:
6
(hidden_1_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:
identityИҐhidden_0/BiasAdd/ReadVariableOpҐhidden_0/MatMul/ReadVariableOpҐhidden_1/BiasAdd/ReadVariableOpҐhidden_1/MatMul/ReadVariableOpҐoutput/MatMul/ReadVariableOpЗ
hidden_0/MatMul/ReadVariableOpReadVariableOp'hidden_0_matmul_readvariableop_resource*
_output_shapes
:	†
*
dtype0{
hidden_0/MatMulMatMulinputs&hidden_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
Д
hidden_0/BiasAdd/ReadVariableOpReadVariableOp(hidden_0_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0С
hidden_0/BiasAddBiasAddhidden_0/MatMul:product:0'hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
R
hidden_0/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?x
hidden_0/mulMulhidden_0/beta:output:0hidden_0/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
_
hidden_0/SigmoidSigmoidhidden_0/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€
x
hidden_0/mul_1Mulhidden_0/BiasAdd:output:0hidden_0/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€
c
hidden_0/IdentityIdentityhidden_0/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€
ƒ
hidden_0/IdentityN	IdentityNhidden_0/mul_1:z:0hidden_0/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-48399*:
_output_shapes(
&:€€€€€€€€€
:€€€€€€€€€
Ж
hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Р
hidden_1/MatMulMatMulhidden_0/IdentityN:output:0&hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€R
hidden_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?x
hidden_1/mulMulhidden_1/beta:output:0hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€_
hidden_1/SigmoidSigmoidhidden_1/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€x
hidden_1/mul_1Mulhidden_1/BiasAdd:output:0hidden_1/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€c
hidden_1/IdentityIdentityhidden_1/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ƒ
hidden_1/IdentityN	IdentityNhidden_1/mul_1:z:0hidden_1/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-48413*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€В
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0М
output/MatMulMatMulhidden_1/IdentityN:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
IdentityIdentityoutput/MatMul:product:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€л
NoOpNoOp ^hidden_0/BiasAdd/ReadVariableOp^hidden_0/MatMul/ReadVariableOp ^hidden_1/BiasAdd/ReadVariableOp^hidden_1/MatMul/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€†: : : : : 2B
hidden_0/BiasAdd/ReadVariableOphidden_0/BiasAdd/ReadVariableOp2@
hidden_0/MatMul/ReadVariableOphidden_0/MatMul/ReadVariableOp2B
hidden_1/BiasAdd/ReadVariableOphidden_1/BiasAdd/ReadVariableOp2@
hidden_1/MatMul/ReadVariableOphidden_1/MatMul/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
Ы
y
"__inference_internal_grad_fn_48640
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:€€€€€€€€€
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€
U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:€€€€€€€€€
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:€€€€€€€€€
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€
"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€
:€€€€€€€€€
: :€€€€€€€€€
:W S
'
_output_shapes
:€€€€€€€€€

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:€€€€€€€€€

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€

√
™
A__inference_output_layer_call_and_return_conditional_losses_48194

inputs0
matmul_readvariableop_resource:
identityИҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
±!
©
E__inference_sequential_layer_call_and_return_conditional_losses_48459

inputs:
'hidden_0_matmul_readvariableop_resource:	†
6
(hidden_0_biasadd_readvariableop_resource:
9
'hidden_1_matmul_readvariableop_resource:
6
(hidden_1_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:
identityИҐhidden_0/BiasAdd/ReadVariableOpҐhidden_0/MatMul/ReadVariableOpҐhidden_1/BiasAdd/ReadVariableOpҐhidden_1/MatMul/ReadVariableOpҐoutput/MatMul/ReadVariableOpЗ
hidden_0/MatMul/ReadVariableOpReadVariableOp'hidden_0_matmul_readvariableop_resource*
_output_shapes
:	†
*
dtype0{
hidden_0/MatMulMatMulinputs&hidden_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
Д
hidden_0/BiasAdd/ReadVariableOpReadVariableOp(hidden_0_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0С
hidden_0/BiasAddBiasAddhidden_0/MatMul:product:0'hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
R
hidden_0/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?x
hidden_0/mulMulhidden_0/beta:output:0hidden_0/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
_
hidden_0/SigmoidSigmoidhidden_0/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€
x
hidden_0/mul_1Mulhidden_0/BiasAdd:output:0hidden_0/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€
c
hidden_0/IdentityIdentityhidden_0/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€
ƒ
hidden_0/IdentityN	IdentityNhidden_0/mul_1:z:0hidden_0/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-48434*:
_output_shapes(
&:€€€€€€€€€
:€€€€€€€€€
Ж
hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Р
hidden_1/MatMulMatMulhidden_0/IdentityN:output:0&hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€R
hidden_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?x
hidden_1/mulMulhidden_1/beta:output:0hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€_
hidden_1/SigmoidSigmoidhidden_1/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€x
hidden_1/mul_1Mulhidden_1/BiasAdd:output:0hidden_1/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€c
hidden_1/IdentityIdentityhidden_1/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ƒ
hidden_1/IdentityN	IdentityNhidden_1/mul_1:z:0hidden_1/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-48448*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€В
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0М
output/MatMulMatMulhidden_1/IdentityN:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
IdentityIdentityoutput/MatMul:product:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€л
NoOpNoOp ^hidden_0/BiasAdd/ReadVariableOp^hidden_0/MatMul/ReadVariableOp ^hidden_1/BiasAdd/ReadVariableOp^hidden_1/MatMul/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€†: : : : : 2B
hidden_0/BiasAdd/ReadVariableOphidden_0/BiasAdd/ReadVariableOp2@
hidden_0/MatMul/ReadVariableOphidden_0/MatMul/ReadVariableOp2B
hidden_1/BiasAdd/ReadVariableOphidden_1/BiasAdd/ReadVariableOp2@
hidden_1/MatMul/ReadVariableOphidden_1/MatMul/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
Х
°
"__inference_internal_grad_fn_48766
result_grads_0
result_grads_1 
mul_sequential_hidden_1_beta#
mul_sequential_hidden_1_biasadd
identityМ
mulMulmul_sequential_hidden_1_betamul_sequential_hidden_1_biasadd^result_grads_0*
T0*'
_output_shapes
:€€€€€€€€€M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€}
mul_1Mulmul_sequential_hidden_1_betamul_sequential_hidden_1_biasadd*
T0*'
_output_shapes
:€€€€€€€€€J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€:€€€€€€€€€: :€€€€€€€€€:W S
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€
ј
Х
(__inference_hidden_1_layer_call_fn_48495

inputs
unknown:

	unknown_0:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_1_layer_call_and_return_conditional_losses_48181o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€

 
_user_specified_nameinputs
і
н
E__inference_sequential_layer_call_and_return_conditional_losses_48319
input_1!
hidden_0_48305:	†

hidden_0_48307:
 
hidden_1_48310:

hidden_1_48312:
output_48315:
identityИҐ hidden_0/StatefulPartitionedCallҐ hidden_1/StatefulPartitionedCallҐoutput/StatefulPartitionedCallо
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinput_1hidden_0_48305hidden_0_48307*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_0_layer_call_and_return_conditional_losses_48157Р
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_48310hidden_1_48312*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_1_layer_call_and_return_conditional_losses_48181ш
output/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0output_48315*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_48194v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≠
NoOpNoOp!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€†: : : : : 2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€†
!
_user_specified_name	input_1
√
™
A__inference_output_layer_call_and_return_conditional_losses_48527

inputs0
matmul_readvariableop_resource:
identityИҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
±
м
E__inference_sequential_layer_call_and_return_conditional_losses_48199

inputs!
hidden_0_48158:	†

hidden_0_48160:
 
hidden_1_48182:

hidden_1_48184:
output_48195:
identityИҐ hidden_0/StatefulPartitionedCallҐ hidden_1/StatefulPartitionedCallҐoutput/StatefulPartitionedCallн
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_0_48158hidden_0_48160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_0_layer_call_and_return_conditional_losses_48157Р
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_48182hidden_1_48184*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_1_layer_call_and_return_conditional_losses_48181ш
output/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0output_48195*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_48194v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≠
NoOpNoOp!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€†: : : : : 2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
і
н
E__inference_sequential_layer_call_and_return_conditional_losses_48336
input_1!
hidden_0_48322:	†

hidden_0_48324:
 
hidden_1_48327:

hidden_1_48329:
output_48332:
identityИҐ hidden_0/StatefulPartitionedCallҐ hidden_1/StatefulPartitionedCallҐoutput/StatefulPartitionedCallо
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinput_1hidden_0_48322hidden_0_48324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_0_layer_call_and_return_conditional_losses_48157Р
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_48327hidden_1_48329*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_1_layer_call_and_return_conditional_losses_48181ш
output/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0output_48332*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_48194v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≠
NoOpNoOp!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€†: : : : : 2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€†
!
_user_specified_name	input_1
У
z
&__inference_output_layer_call_fn_48520

inputs
unknown:
identityИҐStatefulPartitionedCall…
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_48194o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:
"__inference_internal_grad_fn_48604CustomGradient-48505:
"__inference_internal_grad_fn_48622CustomGradient-48173:
"__inference_internal_grad_fn_48640CustomGradient-48478:
"__inference_internal_grad_fn_48658CustomGradient-48149:
"__inference_internal_grad_fn_48676CustomGradient-48434:
"__inference_internal_grad_fn_48694CustomGradient-48448:
"__inference_internal_grad_fn_48712CustomGradient-48399:
"__inference_internal_grad_fn_48730CustomGradient-48413:
"__inference_internal_grad_fn_48748CustomGradient-48107:
"__inference_internal_grad_fn_48766CustomGradient-48121"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*™
serving_defaultЦ
<
input_11
serving_default_input_1:0€€€€€€€€€†:
output0
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:ЌЙ
џ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
ї
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ї
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
±
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel"
_tf_keras_layer
C
0
1
2
3
#4"
trackable_list_wrapper
C
0
1
2
3
#4"
trackable_list_wrapper
 "
trackable_list_wrapper
 
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
Ё
)trace_0
*trace_1
+trace_2
,trace_32т
*__inference_sequential_layer_call_fn_48212
*__inference_sequential_layer_call_fn_48374
*__inference_sequential_layer_call_fn_48389
*__inference_sequential_layer_call_fn_48302њ
ґ≤≤
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

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z)trace_0z*trace_1z+trace_2z,trace_3
…
-trace_0
.trace_1
/trace_2
0trace_32ё
E__inference_sequential_layer_call_and_return_conditional_losses_48424
E__inference_sequential_layer_call_and_return_conditional_losses_48459
E__inference_sequential_layer_call_and_return_conditional_losses_48319
E__inference_sequential_layer_call_and_return_conditional_losses_48336њ
ґ≤≤
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

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z-trace_0z.trace_1z/trace_2z0trace_3
ЋB»
 __inference__wrapped_model_48132input_1"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≠
1iter

2beta_1

3beta_2
	4decay
5learning_ratem]m^m_m`#mavbvcvdve#vf"
	optimizer
,
6serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
м
<trace_02ѕ
(__inference_hidden_0_layer_call_fn_48468Ґ
Щ≤Х
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
annotations™ *
 z<trace_0
З
=trace_02к
C__inference_hidden_0_layer_call_and_return_conditional_losses_48486Ґ
Щ≤Х
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
annotations™ *
 z=trace_0
": 	†
2hidden_0/kernel
:
2hidden_0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
м
Ctrace_02ѕ
(__inference_hidden_1_layer_call_fn_48495Ґ
Щ≤Х
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
annotations™ *
 zCtrace_0
З
Dtrace_02к
C__inference_hidden_1_layer_call_and_return_conditional_losses_48513Ґ
Щ≤Х
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
annotations™ *
 zDtrace_0
!:
2hidden_1/kernel
:2hidden_1/bias
'
#0"
trackable_list_wrapper
'
#0"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
к
Jtrace_02Ќ
&__inference_output_layer_call_fn_48520Ґ
Щ≤Х
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
annotations™ *
 zJtrace_0
Е
Ktrace_02и
A__inference_output_layer_call_and_return_conditional_losses_48527Ґ
Щ≤Х
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
annotations™ *
 zKtrace_0
:2output/kernel
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
L0
M1
N2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ьBщ
*__inference_sequential_layer_call_fn_48212input_1"њ
ґ≤≤
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

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ыBш
*__inference_sequential_layer_call_fn_48374inputs"њ
ґ≤≤
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

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ыBш
*__inference_sequential_layer_call_fn_48389inputs"њ
ґ≤≤
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

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ьBщ
*__inference_sequential_layer_call_fn_48302input_1"њ
ґ≤≤
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

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЦBУ
E__inference_sequential_layer_call_and_return_conditional_losses_48424inputs"њ
ґ≤≤
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

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЦBУ
E__inference_sequential_layer_call_and_return_conditional_losses_48459inputs"њ
ґ≤≤
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

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
E__inference_sequential_layer_call_and_return_conditional_losses_48319input_1"њ
ґ≤≤
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

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
E__inference_sequential_layer_call_and_return_conditional_losses_48336input_1"њ
ґ≤≤
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

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 B«
#__inference_signature_wrapper_48359input_1"Ф
Н≤Й
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
annotations™ *
 
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
№Bў
(__inference_hidden_0_layer_call_fn_48468inputs"Ґ
Щ≤Х
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
annotations™ *
 
чBф
C__inference_hidden_0_layer_call_and_return_conditional_losses_48486inputs"Ґ
Щ≤Х
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
annotations™ *
 
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
№Bў
(__inference_hidden_1_layer_call_fn_48495inputs"Ґ
Щ≤Х
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
annotations™ *
 
чBф
C__inference_hidden_1_layer_call_and_return_conditional_losses_48513inputs"Ґ
Щ≤Х
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
annotations™ *
 
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
ЏB„
&__inference_output_layer_call_fn_48520inputs"Ґ
Щ≤Х
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
annotations™ *
 
хBт
A__inference_output_layer_call_and_return_conditional_losses_48527inputs"Ґ
Щ≤Х
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
annotations™ *
 
N
O	variables
P	keras_api
	Qtotal
	Rcount"
_tf_keras_metric
q
S	variables
T	keras_api
UN
Vsum_squares
Wsum
Xsum_squared_errors"
_tf_keras_metric
N
Y	variables
Z	keras_api
	[total
	\count"
_tf_keras_metric
.
Q0
R1"
trackable_list_wrapper
-
O	variables"
_generic_user_object
:  (2total
:  (2count
<
U0
V1
W2
X3"
trackable_list_wrapper
-
S	variables"
_generic_user_object
: (2N
: (2sum_squares
: (2sum
":  (2sum_squared_errors
.
[0
\1"
trackable_list_wrapper
-
Y	variables"
_generic_user_object
:  (2total
:  (2count
':%	†
2Adam/hidden_0/kernel/m
 :
2Adam/hidden_0/bias/m
&:$
2Adam/hidden_1/kernel/m
 :2Adam/hidden_1/bias/m
$:"2Adam/output/kernel/m
':%	†
2Adam/hidden_0/kernel/v
 :
2Adam/hidden_0/bias/v
&:$
2Adam/hidden_1/kernel/v
 :2Adam/hidden_1/bias/v
$:"2Adam/output/kernel/v
ObM
beta:0C__inference_hidden_1_layer_call_and_return_conditional_losses_48513
RbP
	BiasAdd:0C__inference_hidden_1_layer_call_and_return_conditional_losses_48513
ObM
beta:0C__inference_hidden_1_layer_call_and_return_conditional_losses_48181
RbP
	BiasAdd:0C__inference_hidden_1_layer_call_and_return_conditional_losses_48181
ObM
beta:0C__inference_hidden_0_layer_call_and_return_conditional_losses_48486
RbP
	BiasAdd:0C__inference_hidden_0_layer_call_and_return_conditional_losses_48486
ObM
beta:0C__inference_hidden_0_layer_call_and_return_conditional_losses_48157
RbP
	BiasAdd:0C__inference_hidden_0_layer_call_and_return_conditional_losses_48157
ZbX
hidden_0/beta:0E__inference_sequential_layer_call_and_return_conditional_losses_48459
]b[
hidden_0/BiasAdd:0E__inference_sequential_layer_call_and_return_conditional_losses_48459
ZbX
hidden_1/beta:0E__inference_sequential_layer_call_and_return_conditional_losses_48459
]b[
hidden_1/BiasAdd:0E__inference_sequential_layer_call_and_return_conditional_losses_48459
ZbX
hidden_0/beta:0E__inference_sequential_layer_call_and_return_conditional_losses_48424
]b[
hidden_0/BiasAdd:0E__inference_sequential_layer_call_and_return_conditional_losses_48424
ZbX
hidden_1/beta:0E__inference_sequential_layer_call_and_return_conditional_losses_48424
]b[
hidden_1/BiasAdd:0E__inference_sequential_layer_call_and_return_conditional_losses_48424
@b>
sequential/hidden_0/beta:0 __inference__wrapped_model_48132
CbA
sequential/hidden_0/BiasAdd:0 __inference__wrapped_model_48132
@b>
sequential/hidden_1/beta:0 __inference__wrapped_model_48132
CbA
sequential/hidden_1/BiasAdd:0 __inference__wrapped_model_48132П
 __inference__wrapped_model_48132k#1Ґ.
'Ґ$
"К
input_1€€€€€€€€€†
™ "/™,
*
output К
output€€€€€€€€€§
C__inference_hidden_0_layer_call_and_return_conditional_losses_48486]0Ґ-
&Ґ#
!К
inputs€€€€€€€€€†
™ "%Ґ"
К
0€€€€€€€€€

Ъ |
(__inference_hidden_0_layer_call_fn_48468P0Ґ-
&Ґ#
!К
inputs€€€€€€€€€†
™ "К€€€€€€€€€
£
C__inference_hidden_1_layer_call_and_return_conditional_losses_48513\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€

™ "%Ґ"
К
0€€€€€€€€€
Ъ {
(__inference_hidden_1_layer_call_fn_48495O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€

™ "К€€€€€€€€€Є
"__inference_internal_grad_fn_48604СgheҐb
[ҐX

 
(К%
result_grads_0€€€€€€€€€
(К%
result_grads_1€€€€€€€€€
™ "$Ъ!

 
К
1€€€€€€€€€Є
"__inference_internal_grad_fn_48622СijeҐb
[ҐX

 
(К%
result_grads_0€€€€€€€€€
(К%
result_grads_1€€€€€€€€€
™ "$Ъ!

 
К
1€€€€€€€€€Є
"__inference_internal_grad_fn_48640СkleҐb
[ҐX

 
(К%
result_grads_0€€€€€€€€€

(К%
result_grads_1€€€€€€€€€

™ "$Ъ!

 
К
1€€€€€€€€€
Є
"__inference_internal_grad_fn_48658СmneҐb
[ҐX

 
(К%
result_grads_0€€€€€€€€€

(К%
result_grads_1€€€€€€€€€

™ "$Ъ!

 
К
1€€€€€€€€€
Є
"__inference_internal_grad_fn_48676СopeҐb
[ҐX

 
(К%
result_grads_0€€€€€€€€€

(К%
result_grads_1€€€€€€€€€

™ "$Ъ!

 
К
1€€€€€€€€€
Є
"__inference_internal_grad_fn_48694СqreҐb
[ҐX

 
(К%
result_grads_0€€€€€€€€€
(К%
result_grads_1€€€€€€€€€
™ "$Ъ!

 
К
1€€€€€€€€€Є
"__inference_internal_grad_fn_48712СsteҐb
[ҐX

 
(К%
result_grads_0€€€€€€€€€

(К%
result_grads_1€€€€€€€€€

™ "$Ъ!

 
К
1€€€€€€€€€
Є
"__inference_internal_grad_fn_48730СuveҐb
[ҐX

 
(К%
result_grads_0€€€€€€€€€
(К%
result_grads_1€€€€€€€€€
™ "$Ъ!

 
К
1€€€€€€€€€Є
"__inference_internal_grad_fn_48748СwxeҐb
[ҐX

 
(К%
result_grads_0€€€€€€€€€

(К%
result_grads_1€€€€€€€€€

™ "$Ъ!

 
К
1€€€€€€€€€
Є
"__inference_internal_grad_fn_48766СyzeҐb
[ҐX

 
(К%
result_grads_0€€€€€€€€€
(К%
result_grads_1€€€€€€€€€
™ "$Ъ!

 
К
1€€€€€€€€€†
A__inference_output_layer_call_and_return_conditional_losses_48527[#/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ x
&__inference_output_layer_call_fn_48520N#/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€≤
E__inference_sequential_layer_call_and_return_conditional_losses_48319i#9Ґ6
/Ґ,
"К
input_1€€€€€€€€€†
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ≤
E__inference_sequential_layer_call_and_return_conditional_losses_48336i#9Ґ6
/Ґ,
"К
input_1€€€€€€€€€†
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ±
E__inference_sequential_layer_call_and_return_conditional_losses_48424h#8Ґ5
.Ґ+
!К
inputs€€€€€€€€€†
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ±
E__inference_sequential_layer_call_and_return_conditional_losses_48459h#8Ґ5
.Ґ+
!К
inputs€€€€€€€€€†
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ К
*__inference_sequential_layer_call_fn_48212\#9Ґ6
/Ґ,
"К
input_1€€€€€€€€€†
p 

 
™ "К€€€€€€€€€К
*__inference_sequential_layer_call_fn_48302\#9Ґ6
/Ґ,
"К
input_1€€€€€€€€€†
p

 
™ "К€€€€€€€€€Й
*__inference_sequential_layer_call_fn_48374[#8Ґ5
.Ґ+
!К
inputs€€€€€€€€€†
p 

 
™ "К€€€€€€€€€Й
*__inference_sequential_layer_call_fn_48389[#8Ґ5
.Ґ+
!К
inputs€€€€€€€€€†
p

 
™ "К€€€€€€€€€Э
#__inference_signature_wrapper_48359v#<Ґ9
Ґ 
2™/
-
input_1"К
input_1€€€€€€€€€†"/™,
*
output К
output€€€€€€€€€