Νο
Θ«
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Α
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02unknown8Τ

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

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

Adam/hidden_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/hidden_1/kernel/v

*Adam/hidden_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_1/kernel/v*
_output_shapes

:
*
dtype0

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

Adam/hidden_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 
*'
shared_nameAdam/hidden_0/kernel/v

*Adam/hidden_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_0/kernel/v*
_output_shapes
:	 
*
dtype0

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

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

Adam/hidden_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/hidden_1/kernel/m

*Adam/hidden_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_1/kernel/m*
_output_shapes

:
*
dtype0

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

Adam/hidden_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 
*'
shared_nameAdam/hidden_0/kernel/m

*Adam/hidden_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_0/kernel/m*
_output_shapes
:	 
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
shape:	 
* 
shared_namehidden_0/kernel
t
#hidden_0/kernel/Read/ReadVariableOpReadVariableOphidden_0/kernel*
_output_shapes
:	 
*
dtype0
|
serving_default_input_1Placeholder*(
_output_shapes
:????????? *
dtype0*
shape:????????? 

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1hidden_0/kernelhidden_0/biashidden_1/kernelhidden_1/biasoutput/kernel*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_307251

NoOpNoOp
Τ/
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*/
value/B/ Bϋ.
Α
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
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*

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
°
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

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

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

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

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
|
VARIABLE_VALUEAdam/hidden_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/hidden_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/hidden_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/hidden_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/hidden_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/hidden_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/hidden_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/hidden_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¨

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
GPU 2J 8 *(
f#R!
__inference__traced_save_307706
σ
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_307800σά
Μ
ν
+__inference_sequential_layer_call_fn_307104
input_1
unknown:	 

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_307091o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????? : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:????????? 
!
_user_specified_name	input_1
Ώ
ς
F__inference_sequential_layer_call_and_return_conditional_losses_307166

inputs"
hidden_0_307152:	 

hidden_0_307154:
!
hidden_1_307157:

hidden_1_307159:
output_307162:
identity’ hidden_0/StatefulPartitionedCall’ hidden_1/StatefulPartitionedCall’output/StatefulPartitionedCallπ
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_0_307152hidden_0_307154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_0_layer_call_and_return_conditional_losses_307049
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_307157hidden_1_307159*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_1_layer_call_and_return_conditional_losses_307073ϊ
output/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0output_307162*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_307086v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????­
NoOpNoOp!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????? : : : : : 2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
(
_output_shapes
:????????? 
 
_user_specified_nameinputs
­
ψ
D__inference_hidden_0_layer_call_and_return_conditional_losses_307378

inputs1
matmul_readvariableop_resource:	 
-
biasadd_readvariableop_resource:


identity_1’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:?????????
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????
]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????
Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:?????????
ͺ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-307370*:
_output_shapes(
&:?????????
:?????????
c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:????????? 
 
_user_specified_nameinputs
©
χ
D__inference_hidden_1_layer_call_and_return_conditional_losses_307405

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity_1’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:?????????ͺ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-307397*:
_output_shapes(
&:?????????:?????????c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
Σ

#__inference_internal_grad_fn_307622
result_grads_0
result_grads_1
mul_hidden_1_beta
mul_hidden_1_biasadd
identityv
mulMulmul_hidden_1_betamul_hidden_1_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????g
mul_1Mulmul_hidden_1_betamul_hidden_1_biasadd*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????:?????????: :?????????:W S
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????
Ι
μ
+__inference_sequential_layer_call_fn_307266

inputs
unknown:	 

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_307091o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????? : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:????????? 
 
_user_specified_nameinputs
Δ
«
B__inference_output_layer_call_and_return_conditional_losses_307086

inputs0
matmul_readvariableop_resource:
identity’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Σ

#__inference_internal_grad_fn_307604
result_grads_0
result_grads_1
mul_hidden_0_beta
mul_hidden_0_biasadd
identityv
mulMulmul_hidden_0_betamul_hidden_0_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????
g
mul_1Mulmul_hidden_0_betamul_hidden_0_biasadd*
T0*'
_output_shapes
:?????????
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????
:?????????
: :?????????
:W S
'
_output_shapes
:?????????

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????

Ε

)__inference_hidden_0_layer_call_fn_307360

inputs
unknown:	 

	unknown_0:

identity’StatefulPartitionedCallΩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_0_layer_call_and_return_conditional_losses_307049o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:????????? 
 
_user_specified_nameinputs

z
#__inference_internal_grad_fn_307496
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????:?????????: :?????????:W S
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????
Β
σ
F__inference_sequential_layer_call_and_return_conditional_losses_307211
input_1"
hidden_0_307197:	 

hidden_0_307199:
!
hidden_1_307202:

hidden_1_307204:
output_307207:
identity’ hidden_0/StatefulPartitionedCall’ hidden_1/StatefulPartitionedCall’output/StatefulPartitionedCallρ
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinput_1hidden_0_307197hidden_0_307199*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_0_layer_call_and_return_conditional_losses_307049
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_307202hidden_1_307204*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_1_layer_call_and_return_conditional_losses_307073ϊ
output/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0output_307207*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_307086v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????­
NoOpNoOp!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????? : : : : : 2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Q M
(
_output_shapes
:????????? 
!
_user_specified_name	input_1

z
#__inference_internal_grad_fn_307532
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????
U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:?????????
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????
:?????????
: :?????????
:W S
'
_output_shapes
:?????????

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????

 
ζ
$__inference_signature_wrapper_307251
input_1
unknown:	 

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
identity’StatefulPartitionedCallή
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_307024o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????? : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:????????? 
!
_user_specified_name	input_1

{
'__inference_output_layer_call_fn_307412

inputs
unknown:
identity’StatefulPartitionedCallΚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_307086o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Ώ
ς
F__inference_sequential_layer_call_and_return_conditional_losses_307091

inputs"
hidden_0_307050:	 

hidden_0_307052:
!
hidden_1_307074:

hidden_1_307076:
output_307087:
identity’ hidden_0/StatefulPartitionedCall’ hidden_1/StatefulPartitionedCall’output/StatefulPartitionedCallπ
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_0_307050hidden_0_307052*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_0_layer_call_and_return_conditional_losses_307049
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_307074hidden_1_307076*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_1_layer_call_and_return_conditional_losses_307073ϊ
output/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0output_307087*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_307086v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????­
NoOpNoOp!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????? : : : : : 2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
(
_output_shapes
:????????? 
 
_user_specified_nameinputs
Β

)__inference_hidden_1_layer_call_fn_307387

inputs
unknown:

	unknown_0:
identity’StatefulPartitionedCallΩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_1_layer_call_and_return_conditional_losses_307073o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
©
χ
D__inference_hidden_1_layer_call_and_return_conditional_losses_307073

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity_1’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:?????????ͺ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-307065*:
_output_shapes(
&:?????????:?????????c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
Β
σ
F__inference_sequential_layer_call_and_return_conditional_losses_307228
input_1"
hidden_0_307214:	 

hidden_0_307216:
!
hidden_1_307219:

hidden_1_307221:
output_307224:
identity’ hidden_0/StatefulPartitionedCall’ hidden_1/StatefulPartitionedCall’output/StatefulPartitionedCallρ
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinput_1hidden_0_307214hidden_0_307216*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_0_layer_call_and_return_conditional_losses_307049
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_307219hidden_1_307221*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_1_layer_call_and_return_conditional_losses_307073ϊ
output/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0output_307224*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_307086v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????­
NoOpNoOp!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????? : : : : : 2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Q M
(
_output_shapes
:????????? 
!
_user_specified_name	input_1

z
#__inference_internal_grad_fn_307514
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????:?????????: :?????????:W S
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????

’
#__inference_internal_grad_fn_307658
result_grads_0
result_grads_1 
mul_sequential_hidden_1_beta#
mul_sequential_hidden_1_biasadd
identity
mulMulmul_sequential_hidden_1_betamul_sequential_hidden_1_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????}
mul_1Mulmul_sequential_hidden_1_betamul_sequential_hidden_1_biasadd*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????:?????????: :?????????:W S
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????
­
ψ
D__inference_hidden_0_layer_call_and_return_conditional_losses_307049

inputs1
matmul_readvariableop_resource:	 
-
biasadd_readvariableop_resource:


identity_1’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:?????????
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????
]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????
Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:?????????
ͺ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-307041*:
_output_shapes(
&:?????????
:?????????
c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:????????? 
 
_user_specified_nameinputs
Σ

#__inference_internal_grad_fn_307586
result_grads_0
result_grads_1
mul_hidden_1_beta
mul_hidden_1_biasadd
identityv
mulMulmul_hidden_1_betamul_hidden_1_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????g
mul_1Mulmul_hidden_1_betamul_hidden_1_biasadd*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????:?????????: :?????????:W S
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????
Δ
«
B__inference_output_layer_call_and_return_conditional_losses_307419

inputs0
matmul_readvariableop_resource:
identity’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ρ'
τ
!__inference__wrapped_model_307024
input_1E
2sequential_hidden_0_matmul_readvariableop_resource:	 
A
3sequential_hidden_0_biasadd_readvariableop_resource:
D
2sequential_hidden_1_matmul_readvariableop_resource:
A
3sequential_hidden_1_biasadd_readvariableop_resource:B
0sequential_output_matmul_readvariableop_resource:
identity’*sequential/hidden_0/BiasAdd/ReadVariableOp’)sequential/hidden_0/MatMul/ReadVariableOp’*sequential/hidden_1/BiasAdd/ReadVariableOp’)sequential/hidden_1/MatMul/ReadVariableOp’'sequential/output/MatMul/ReadVariableOp
)sequential/hidden_0/MatMul/ReadVariableOpReadVariableOp2sequential_hidden_0_matmul_readvariableop_resource*
_output_shapes
:	 
*
dtype0
sequential/hidden_0/MatMulMatMulinput_11sequential/hidden_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????

*sequential/hidden_0/BiasAdd/ReadVariableOpReadVariableOp3sequential_hidden_0_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0²
sequential/hidden_0/BiasAddBiasAdd$sequential/hidden_0/MatMul:product:02sequential/hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
]
sequential/hidden_0/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
sequential/hidden_0/mulMul!sequential/hidden_0/beta:output:0$sequential/hidden_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
u
sequential/hidden_0/SigmoidSigmoidsequential/hidden_0/mul:z:0*
T0*'
_output_shapes
:?????????

sequential/hidden_0/mul_1Mul$sequential/hidden_0/BiasAdd:output:0sequential/hidden_0/Sigmoid:y:0*
T0*'
_output_shapes
:?????????
y
sequential/hidden_0/IdentityIdentitysequential/hidden_0/mul_1:z:0*
T0*'
_output_shapes
:?????????
ζ
sequential/hidden_0/IdentityN	IdentityNsequential/hidden_0/mul_1:z:0$sequential/hidden_0/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-306999*:
_output_shapes(
&:?????????
:?????????

)sequential/hidden_1/MatMul/ReadVariableOpReadVariableOp2sequential_hidden_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0±
sequential/hidden_1/MatMulMatMul&sequential/hidden_0/IdentityN:output:01sequential/hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
*sequential/hidden_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_hidden_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0²
sequential/hidden_1/BiasAddBiasAdd$sequential/hidden_1/MatMul:product:02sequential/hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????]
sequential/hidden_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
sequential/hidden_1/mulMul!sequential/hidden_1/beta:output:0$sequential/hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????u
sequential/hidden_1/SigmoidSigmoidsequential/hidden_1/mul:z:0*
T0*'
_output_shapes
:?????????
sequential/hidden_1/mul_1Mul$sequential/hidden_1/BiasAdd:output:0sequential/hidden_1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????y
sequential/hidden_1/IdentityIdentitysequential/hidden_1/mul_1:z:0*
T0*'
_output_shapes
:?????????ζ
sequential/hidden_1/IdentityN	IdentityNsequential/hidden_1/mul_1:z:0$sequential/hidden_1/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-307013*:
_output_shapes(
&:?????????:?????????
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0­
sequential/output/MatMulMatMul&sequential/hidden_1/IdentityN:output:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
IdentityIdentity"sequential/output/MatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????’
NoOpNoOp+^sequential/hidden_0/BiasAdd/ReadVariableOp*^sequential/hidden_0/MatMul/ReadVariableOp+^sequential/hidden_1/BiasAdd/ReadVariableOp*^sequential/hidden_1/MatMul/ReadVariableOp(^sequential/output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????? : : : : : 2X
*sequential/hidden_0/BiasAdd/ReadVariableOp*sequential/hidden_0/BiasAdd/ReadVariableOp2V
)sequential/hidden_0/MatMul/ReadVariableOp)sequential/hidden_0/MatMul/ReadVariableOp2X
*sequential/hidden_1/BiasAdd/ReadVariableOp*sequential/hidden_1/BiasAdd/ReadVariableOp2V
)sequential/hidden_1/MatMul/ReadVariableOp)sequential/hidden_1/MatMul/ReadVariableOp2R
'sequential/output/MatMul/ReadVariableOp'sequential/output/MatMul/ReadVariableOp:Q M
(
_output_shapes
:????????? 
!
_user_specified_name	input_1
§o

"__inference__traced_restore_307800
file_prefix3
 assignvariableop_hidden_0_kernel:	 
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
*assignvariableop_18_adam_hidden_0_kernel_m:	 
6
(assignvariableop_19_adam_hidden_0_bias_m:
<
*assignvariableop_20_adam_hidden_1_kernel_m:
6
(assignvariableop_21_adam_hidden_1_bias_m::
(assignvariableop_22_adam_output_kernel_m:=
*assignvariableop_23_adam_hidden_0_kernel_v:	 
6
(assignvariableop_24_adam_hidden_0_bias_v:
<
*assignvariableop_25_adam_hidden_1_kernel_v:
6
(assignvariableop_26_adam_hidden_1_bias_v::
(assignvariableop_27_adam_output_kernel_v:
identity_29’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*©
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB0keras_api/metrics/1/N/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/1/sum_squares/.ATTRIBUTES/VARIABLE_VALUEB2keras_api/metrics/1/sum/.ATTRIBUTES/VARIABLE_VALUEBAkeras_api/metrics/1/sum_squared_errors/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHͺ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B °
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_hidden_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_hidden_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_hidden_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_hidden_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_nIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_sum_squaresIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_sumIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp&assignvariableop_15_sum_squared_errorsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_hidden_0_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_hidden_0_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_hidden_1_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_hidden_1_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_output_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_hidden_0_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_hidden_0_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_hidden_1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_hidden_1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_output_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ·
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: €
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
Μ
ν
+__inference_sequential_layer_call_fn_307194
input_1
unknown:	 

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_307166o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????? : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:????????? 
!
_user_specified_name	input_1

’
#__inference_internal_grad_fn_307640
result_grads_0
result_grads_1 
mul_sequential_hidden_0_beta#
mul_sequential_hidden_0_biasadd
identity
mulMulmul_sequential_hidden_0_betamul_sequential_hidden_0_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????
}
mul_1Mulmul_sequential_hidden_0_betamul_sequential_hidden_0_biasadd*
T0*'
_output_shapes
:?????????
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????
:?????????
: :?????????
:W S
'
_output_shapes
:?????????

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????

Σ

#__inference_internal_grad_fn_307568
result_grads_0
result_grads_1
mul_hidden_0_beta
mul_hidden_0_biasadd
identityv
mulMulmul_hidden_0_betamul_hidden_0_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????
g
mul_1Mulmul_hidden_0_betamul_hidden_0_biasadd*
T0*'
_output_shapes
:?????????
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????
:?????????
: :?????????
:W S
'
_output_shapes
:?????????

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????

΄!
ͺ
F__inference_sequential_layer_call_and_return_conditional_losses_307316

inputs:
'hidden_0_matmul_readvariableop_resource:	 
6
(hidden_0_biasadd_readvariableop_resource:
9
'hidden_1_matmul_readvariableop_resource:
6
(hidden_1_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:
identity’hidden_0/BiasAdd/ReadVariableOp’hidden_0/MatMul/ReadVariableOp’hidden_1/BiasAdd/ReadVariableOp’hidden_1/MatMul/ReadVariableOp’output/MatMul/ReadVariableOp
hidden_0/MatMul/ReadVariableOpReadVariableOp'hidden_0_matmul_readvariableop_resource*
_output_shapes
:	 
*
dtype0{
hidden_0/MatMulMatMulinputs&hidden_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????

hidden_0/BiasAdd/ReadVariableOpReadVariableOp(hidden_0_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
hidden_0/BiasAddBiasAddhidden_0/MatMul:product:0'hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
R
hidden_0/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
hidden_0/mulMulhidden_0/beta:output:0hidden_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
_
hidden_0/SigmoidSigmoidhidden_0/mul:z:0*
T0*'
_output_shapes
:?????????
x
hidden_0/mul_1Mulhidden_0/BiasAdd:output:0hidden_0/Sigmoid:y:0*
T0*'
_output_shapes
:?????????
c
hidden_0/IdentityIdentityhidden_0/mul_1:z:0*
T0*'
_output_shapes
:?????????
Ε
hidden_0/IdentityN	IdentityNhidden_0/mul_1:z:0hidden_0/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-307291*:
_output_shapes(
&:?????????
:?????????

hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
hidden_1/MatMulMatMulhidden_0/IdentityN:output:0&hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????R
hidden_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
hidden_1/mulMulhidden_1/beta:output:0hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????_
hidden_1/SigmoidSigmoidhidden_1/mul:z:0*
T0*'
_output_shapes
:?????????x
hidden_1/mul_1Mulhidden_1/BiasAdd:output:0hidden_1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????c
hidden_1/IdentityIdentityhidden_1/mul_1:z:0*
T0*'
_output_shapes
:?????????Ε
hidden_1/IdentityN	IdentityNhidden_1/mul_1:z:0hidden_1/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-307305*:
_output_shapes(
&:?????????:?????????
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
output/MatMulMatMulhidden_1/IdentityN:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
IdentityIdentityoutput/MatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????λ
NoOpNoOp ^hidden_0/BiasAdd/ReadVariableOp^hidden_0/MatMul/ReadVariableOp ^hidden_1/BiasAdd/ReadVariableOp^hidden_1/MatMul/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????? : : : : : 2B
hidden_0/BiasAdd/ReadVariableOphidden_0/BiasAdd/ReadVariableOp2@
hidden_0/MatMul/ReadVariableOphidden_0/MatMul/ReadVariableOp2B
hidden_1/BiasAdd/ReadVariableOphidden_1/BiasAdd/ReadVariableOp2@
hidden_1/MatMul/ReadVariableOphidden_1/MatMul/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:P L
(
_output_shapes
:????????? 
 
_user_specified_nameinputs
 <

__inference__traced_save_307706
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

identity_1’MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*©
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB0keras_api/metrics/1/N/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/1/sum_squares/.ATTRIBUTES/VARIABLE_VALUEB2keras_api/metrics/1/sum/.ATTRIBUTES/VARIABLE_VALUEBAkeras_api/metrics/1/sum_squared_errors/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH§
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_hidden_0_kernel_read_readvariableop(savev2_hidden_0_bias_read_readvariableop*savev2_hidden_1_kernel_read_readvariableop(savev2_hidden_1_bias_read_readvariableop(savev2_output_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_n_read_readvariableop&savev2_sum_squares_read_readvariableopsavev2_sum_read_readvariableop-savev2_sum_squared_errors_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_hidden_0_kernel_m_read_readvariableop/savev2_adam_hidden_0_bias_m_read_readvariableop1savev2_adam_hidden_1_kernel_m_read_readvariableop/savev2_adam_hidden_1_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop1savev2_adam_hidden_0_kernel_v_read_readvariableop/savev2_adam_hidden_0_bias_v_read_readvariableop1savev2_adam_hidden_1_kernel_v_read_readvariableop/savev2_adam_hidden_1_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*Δ
_input_shapes²
―: :	 
:
:
::: : : : : : : ::::: : :	 
:
:
:::	 
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
:	 
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
:	 
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
:	 
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

z
#__inference_internal_grad_fn_307550
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:?????????
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:?????????
U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:?????????
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:?????????
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:?????????
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:?????????
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:?????????
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????
:?????????
: :?????????
:W S
'
_output_shapes
:?????????

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:?????????

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:?????????

΄!
ͺ
F__inference_sequential_layer_call_and_return_conditional_losses_307351

inputs:
'hidden_0_matmul_readvariableop_resource:	 
6
(hidden_0_biasadd_readvariableop_resource:
9
'hidden_1_matmul_readvariableop_resource:
6
(hidden_1_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:
identity’hidden_0/BiasAdd/ReadVariableOp’hidden_0/MatMul/ReadVariableOp’hidden_1/BiasAdd/ReadVariableOp’hidden_1/MatMul/ReadVariableOp’output/MatMul/ReadVariableOp
hidden_0/MatMul/ReadVariableOpReadVariableOp'hidden_0_matmul_readvariableop_resource*
_output_shapes
:	 
*
dtype0{
hidden_0/MatMulMatMulinputs&hidden_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????

hidden_0/BiasAdd/ReadVariableOpReadVariableOp(hidden_0_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
hidden_0/BiasAddBiasAddhidden_0/MatMul:product:0'hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
R
hidden_0/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
hidden_0/mulMulhidden_0/beta:output:0hidden_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
_
hidden_0/SigmoidSigmoidhidden_0/mul:z:0*
T0*'
_output_shapes
:?????????
x
hidden_0/mul_1Mulhidden_0/BiasAdd:output:0hidden_0/Sigmoid:y:0*
T0*'
_output_shapes
:?????????
c
hidden_0/IdentityIdentityhidden_0/mul_1:z:0*
T0*'
_output_shapes
:?????????
Ε
hidden_0/IdentityN	IdentityNhidden_0/mul_1:z:0hidden_0/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-307326*:
_output_shapes(
&:?????????
:?????????

hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
hidden_1/MatMulMatMulhidden_0/IdentityN:output:0&hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????R
hidden_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
hidden_1/mulMulhidden_1/beta:output:0hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????_
hidden_1/SigmoidSigmoidhidden_1/mul:z:0*
T0*'
_output_shapes
:?????????x
hidden_1/mul_1Mulhidden_1/BiasAdd:output:0hidden_1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????c
hidden_1/IdentityIdentityhidden_1/mul_1:z:0*
T0*'
_output_shapes
:?????????Ε
hidden_1/IdentityN	IdentityNhidden_1/mul_1:z:0hidden_1/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-307340*:
_output_shapes(
&:?????????:?????????
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
output/MatMulMatMulhidden_1/IdentityN:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
IdentityIdentityoutput/MatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????λ
NoOpNoOp ^hidden_0/BiasAdd/ReadVariableOp^hidden_0/MatMul/ReadVariableOp ^hidden_1/BiasAdd/ReadVariableOp^hidden_1/MatMul/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????? : : : : : 2B
hidden_0/BiasAdd/ReadVariableOphidden_0/BiasAdd/ReadVariableOp2@
hidden_0/MatMul/ReadVariableOphidden_0/MatMul/ReadVariableOp2B
hidden_1/BiasAdd/ReadVariableOphidden_1/BiasAdd/ReadVariableOp2@
hidden_1/MatMul/ReadVariableOphidden_1/MatMul/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:P L
(
_output_shapes
:????????? 
 
_user_specified_nameinputs
Ι
μ
+__inference_sequential_layer_call_fn_307281

inputs
unknown:	 

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_307166o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????? : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:????????? 
 
_user_specified_nameinputs<
#__inference_internal_grad_fn_307496CustomGradient-307397<
#__inference_internal_grad_fn_307514CustomGradient-307065<
#__inference_internal_grad_fn_307532CustomGradient-307370<
#__inference_internal_grad_fn_307550CustomGradient-307041<
#__inference_internal_grad_fn_307568CustomGradient-307326<
#__inference_internal_grad_fn_307586CustomGradient-307340<
#__inference_internal_grad_fn_307604CustomGradient-307291<
#__inference_internal_grad_fn_307622CustomGradient-307305<
#__inference_internal_grad_fn_307640CustomGradient-306999<
#__inference_internal_grad_fn_307658CustomGradient-307013"΅	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ͺ
serving_default
<
input_11
serving_default_input_1:0????????? :
output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:
Ϋ
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
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
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
Κ
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
α
)trace_0
*trace_1
+trace_2
,trace_32φ
+__inference_sequential_layer_call_fn_307104
+__inference_sequential_layer_call_fn_307266
+__inference_sequential_layer_call_fn_307281
+__inference_sequential_layer_call_fn_307194Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z)trace_0z*trace_1z+trace_2z,trace_3
Ν
-trace_0
.trace_1
/trace_2
0trace_32β
F__inference_sequential_layer_call_and_return_conditional_losses_307316
F__inference_sequential_layer_call_and_return_conditional_losses_307351
F__inference_sequential_layer_call_and_return_conditional_losses_307211
F__inference_sequential_layer_call_and_return_conditional_losses_307228Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z-trace_0z.trace_1z/trace_2z0trace_3
ΜBΙ
!__inference__wrapped_model_307024input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
­
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
­
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
ν
<trace_02Π
)__inference_hidden_0_layer_call_fn_307360’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z<trace_0

=trace_02λ
D__inference_hidden_0_layer_call_and_return_conditional_losses_307378’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z=trace_0
": 	 
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
­
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
ν
Ctrace_02Π
)__inference_hidden_1_layer_call_fn_307387’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zCtrace_0

Dtrace_02λ
D__inference_hidden_1_layer_call_and_return_conditional_losses_307405’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
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
­
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
λ
Jtrace_02Ξ
'__inference_output_layer_call_fn_307412’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zJtrace_0

Ktrace_02ι
B__inference_output_layer_call_and_return_conditional_losses_307419’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
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
ύBϊ
+__inference_sequential_layer_call_fn_307104input_1"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
όBω
+__inference_sequential_layer_call_fn_307266inputs"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
όBω
+__inference_sequential_layer_call_fn_307281inputs"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ύBϊ
+__inference_sequential_layer_call_fn_307194input_1"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_307316inputs"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_307351inputs"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_307211input_1"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_307228input_1"Ώ
Ά²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ΛBΘ
$__inference_signature_wrapper_307251input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
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
έBΪ
)__inference_hidden_0_layer_call_fn_307360inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ψBυ
D__inference_hidden_0_layer_call_and_return_conditional_losses_307378inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
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
έBΪ
)__inference_hidden_1_layer_call_fn_307387inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ψBυ
D__inference_hidden_1_layer_call_and_return_conditional_losses_307405inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
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
ΫBΨ
'__inference_output_layer_call_fn_307412inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
φBσ
B__inference_output_layer_call_and_return_conditional_losses_307419inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
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
':%	 
2Adam/hidden_0/kernel/m
 :
2Adam/hidden_0/bias/m
&:$
2Adam/hidden_1/kernel/m
 :2Adam/hidden_1/bias/m
$:"2Adam/output/kernel/m
':%	 
2Adam/hidden_0/kernel/v
 :
2Adam/hidden_0/bias/v
&:$
2Adam/hidden_1/kernel/v
 :2Adam/hidden_1/bias/v
$:"2Adam/output/kernel/v
PbN
beta:0D__inference_hidden_1_layer_call_and_return_conditional_losses_307405
SbQ
	BiasAdd:0D__inference_hidden_1_layer_call_and_return_conditional_losses_307405
PbN
beta:0D__inference_hidden_1_layer_call_and_return_conditional_losses_307073
SbQ
	BiasAdd:0D__inference_hidden_1_layer_call_and_return_conditional_losses_307073
PbN
beta:0D__inference_hidden_0_layer_call_and_return_conditional_losses_307378
SbQ
	BiasAdd:0D__inference_hidden_0_layer_call_and_return_conditional_losses_307378
PbN
beta:0D__inference_hidden_0_layer_call_and_return_conditional_losses_307049
SbQ
	BiasAdd:0D__inference_hidden_0_layer_call_and_return_conditional_losses_307049
[bY
hidden_0/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_307351
^b\
hidden_0/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_307351
[bY
hidden_1/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_307351
^b\
hidden_1/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_307351
[bY
hidden_0/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_307316
^b\
hidden_0/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_307316
[bY
hidden_1/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_307316
^b\
hidden_1/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_307316
Ab?
sequential/hidden_0/beta:0!__inference__wrapped_model_307024
DbB
sequential/hidden_0/BiasAdd:0!__inference__wrapped_model_307024
Ab?
sequential/hidden_1/beta:0!__inference__wrapped_model_307024
DbB
sequential/hidden_1/BiasAdd:0!__inference__wrapped_model_307024
!__inference__wrapped_model_307024k#1’.
'’$
"
input_1????????? 
ͺ "/ͺ,
*
output 
output?????????₯
D__inference_hidden_0_layer_call_and_return_conditional_losses_307378]0’-
&’#
!
inputs????????? 
ͺ "%’"

0?????????

 }
)__inference_hidden_0_layer_call_fn_307360P0’-
&’#
!
inputs????????? 
ͺ "?????????
€
D__inference_hidden_1_layer_call_and_return_conditional_losses_307405\/’,
%’"
 
inputs?????????

ͺ "%’"

0?????????
 |
)__inference_hidden_1_layer_call_fn_307387O/’,
%’"
 
inputs?????????

ͺ "?????????Ή
#__inference_internal_grad_fn_307496ghe’b
[’X

 
(%
result_grads_0?????????
(%
result_grads_1?????????
ͺ "$!

 

1?????????Ή
#__inference_internal_grad_fn_307514ije’b
[’X

 
(%
result_grads_0?????????
(%
result_grads_1?????????
ͺ "$!

 

1?????????Ή
#__inference_internal_grad_fn_307532kle’b
[’X

 
(%
result_grads_0?????????

(%
result_grads_1?????????

ͺ "$!

 

1?????????
Ή
#__inference_internal_grad_fn_307550mne’b
[’X

 
(%
result_grads_0?????????

(%
result_grads_1?????????

ͺ "$!

 

1?????????
Ή
#__inference_internal_grad_fn_307568ope’b
[’X

 
(%
result_grads_0?????????

(%
result_grads_1?????????

ͺ "$!

 

1?????????
Ή
#__inference_internal_grad_fn_307586qre’b
[’X

 
(%
result_grads_0?????????
(%
result_grads_1?????????
ͺ "$!

 

1?????????Ή
#__inference_internal_grad_fn_307604ste’b
[’X

 
(%
result_grads_0?????????

(%
result_grads_1?????????

ͺ "$!

 

1?????????
Ή
#__inference_internal_grad_fn_307622uve’b
[’X

 
(%
result_grads_0?????????
(%
result_grads_1?????????
ͺ "$!

 

1?????????Ή
#__inference_internal_grad_fn_307640wxe’b
[’X

 
(%
result_grads_0?????????

(%
result_grads_1?????????

ͺ "$!

 

1?????????
Ή
#__inference_internal_grad_fn_307658yze’b
[’X

 
(%
result_grads_0?????????
(%
result_grads_1?????????
ͺ "$!

 

1?????????‘
B__inference_output_layer_call_and_return_conditional_losses_307419[#/’,
%’"
 
inputs?????????
ͺ "%’"

0?????????
 y
'__inference_output_layer_call_fn_307412N#/’,
%’"
 
inputs?????????
ͺ "?????????³
F__inference_sequential_layer_call_and_return_conditional_losses_307211i#9’6
/’,
"
input_1????????? 
p 

 
ͺ "%’"

0?????????
 ³
F__inference_sequential_layer_call_and_return_conditional_losses_307228i#9’6
/’,
"
input_1????????? 
p

 
ͺ "%’"

0?????????
 ²
F__inference_sequential_layer_call_and_return_conditional_losses_307316h#8’5
.’+
!
inputs????????? 
p 

 
ͺ "%’"

0?????????
 ²
F__inference_sequential_layer_call_and_return_conditional_losses_307351h#8’5
.’+
!
inputs????????? 
p

 
ͺ "%’"

0?????????
 
+__inference_sequential_layer_call_fn_307104\#9’6
/’,
"
input_1????????? 
p 

 
ͺ "?????????
+__inference_sequential_layer_call_fn_307194\#9’6
/’,
"
input_1????????? 
p

 
ͺ "?????????
+__inference_sequential_layer_call_fn_307266[#8’5
.’+
!
inputs????????? 
p 

 
ͺ "?????????
+__inference_sequential_layer_call_fn_307281[#8’5
.’+
!
inputs????????? 
p

 
ͺ "?????????
$__inference_signature_wrapper_307251v#<’9
’ 
2ͺ/
-
input_1"
input_1????????? "/ͺ,
*
output 
output?????????