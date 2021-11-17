import numpy as np
import streamlit as st
import plotly.express as px

import plotly.graph_objects as go
from matplotlib import pyplot as plt

st.set_page_config( layout='wide' )

class Grad( object ):
	'''
	__init__() start functions to use
		- linreg -> linear compute function.
		- loss   -> loss function to gradient.
		- gradient -> compute the gradient.
	set_html()
		- Load html and css for streamlit app.
	geral_gradient()
		- Compute gradient with loop and return dict with informations.
	plot_charts()
		- Show gradient steps and other plots.
	'''
	def __init__( self ):
		self.linreg   = lambda x, a=1, b=1: a * x + b
		self.loss     = lambda ytrue, yhat: .5 * ( ytrue - yhat ) * ( ytrue - yhat )
		self.gradient = lambda ytrue, yhat, x=1: - ( ytrue - yhat ) * x

	def set_html( self ):
		
		html1 = '''<h1>❏ Simple Gradient Model</h1>'''
		st.markdown( html1, unsafe_allow_html=True )
		st.write(' ')
		
		st.header('➯ What is Gradient Descent?')
		st.sidebar.title('❏ Geral Filters')
		st.write(' ')

		html2 = '''
		<style>

		h1 {
			color: #7647ff;
			text-align:center
		}

		::selection {
			color: #b950ff;
		}

		h2{
			color: #bca6ff;
			transition: linear 1s;
		}

		h2:hover {
			color: red;
		}

		.bar {
			position: absolute;
			top: 30px;
			width: 200px;
			height: 5px;
			background: white;
			transition: linear 1s;
		}

		.bar:hover {
			background: red;
		}

		.bar2 {
			position: absolute;
			top: 50px;
			width: 150px;
			height: 5px;
			background: red;
			transition: linear 1s;
		}

		.bar2:hover {
			background: white;
		}

		.bar3{
			position: absolute;
			top: 70px;
			width: 100px;
			height: 5px;
			background: white;
			transition: linear 1s;
		}

		.bar3:hover {
			background: red;
		}

		.bar4 {
			position: absolute;
			top: 90px;
			width: 50px;
			height: 5px;
			background: red;
			transition: linear 1s;

		}

		.bar4:hover {
			background: white;
		}

		.par {
			margin: 60px 100px;
			font-size: 20px;
			font-family: fangsong;
			padding: 50px;
			width: 50%;
			box-shadow: 3px 3px 3px white
		}

		@media screen and (max-width: 800px) {
			.par {
				width: 75%;
			}
		}

		</style>
		<section style='text-align: center;'>
			<div class='bar'></div>
			<div class='bar2'></div>
			<div class='bar3'></div>
			<div class='bar4'></div>
			<p class='par'>The Gradient Descent is an optimization algorithm, it is possible to optimize a function to calculate the maximum or the minimum point. Maximize profit or minimize expense.</p>
			</section>
		'''
		st.markdown( html2, unsafe_allow_html=True )
		st.write(' ')
		
		return None
		

	def geral_gradient( self ):
		st.sidebar.header('Linear Coefficients')
		A = st.sidebar.slider( 'Select Coef "A"', -20, 20, 5 )
		B = st.sidebar.slider( 'Select Coef "B"', -50, 50, -20 )

		st.sidebar.header('Gradient Parameters')
		disp     = st.sidebar.slider( 'Select Dispersion', 1, 30, 7 )
		eta1     = st.sidebar.slider( 'Select Learning Rate 1', .001, .1, .01)
		eta2     = st.sidebar.slider( 'Select Learning Rate 2', .1, .90, .5)
		ninter   = st.sidebar.slider( 'Select Max Number of Interactions', 5, 200, 20)
		n_points = st.slider( 'Number of Negative Points', -25, 0, -20 )
		p_points = st.slider( 'Number of Positive Points', 0, 25, 20 )

		x = np.arange( n_points, p_points, 1 )
		y = A * x + B + np.random.normal( 0, disp, len(x) )
		n = len(x)

		errors    = []
		slope     = []
		intercept = []
		gradA     = []
		gradB     = []

		i = 0
		a, b = 1, 0
		while i < ninter:
			error = 0
			da = 0
			db = 0
			
			for xi, yi in zip(x, y):
				yhat = self.linreg( xi, a, b )
				error += self.loss( yi, yhat ) / n
				da += self.gradient( yi, yhat, xi ) / n
				db += self.gradient( yi, yhat, 1 ) / n
				
			a = a - eta1 * da
			b = b - eta2 * db

			slope.append( a )
			gradA.append( da )
			gradB.append( db )
			intercept.append( b )
			errors.append( error )

			i += 1

		errors    = np.array( errors )
		slope     = np.array( slope )
		intercept = np.array( intercept )
		gradA     = np.array( gradA ) 
		gradB     = np.array( gradB )
		anew      = np.arange( A, 50, 1 )
		bnew      = np.arange( B, 50, 1 )
		enew      = np.arange( 0, 50, 1 )
		
		info = {'x':x, 
			    'y': y,
			    'A':A,
			    'B':B,
			    'errors':errors,
			    'slope':slope,
			    'intercept':intercept,
			    'gradA':gradA,
			    'gradB':gradB,
			    'anew':anew,
			    'bnew':bnew,
			    'enew':enew}

		return info

	def plot_charts( self, x, y, A, B, errors, slope, intercept, gradA, gradB, anew, bnew, enew ):
		
		fig = px.scatter( x=x, y=y, trendline='ols' )
		st.plotly_chart( fig, use_container_width=True )

		html2 = '''<h2>➯ (Slope) Model A</h2>'''
		st.markdown( html2, unsafe_allow_html=True )
		
		fig = go.Figure(data=go.Scatter(x=anew, y=slope))
		fig.update_layout( plot_bgcolor='#0e1117' )
		st.plotly_chart( fig, use_container_width=True )


		html2 = '''<h2>➯ (Intercept) Model B</h2>'''
		st.markdown( html2, unsafe_allow_html=True )
		
		fig = go.Figure(data=go.Scatter(x=bnew, y=intercept))
		fig.update_layout( plot_bgcolor='#0e1117' )
		st.plotly_chart( fig, use_container_width=True )

		html2 = '''<h2>✘ Geral Errors</h2>'''
		st.markdown( html2, unsafe_allow_html=True )
		
		fig = go.Figure(data=go.Scatter(x=enew, y=errors))
		fig.update_layout( plot_bgcolor='#0e1117' )
		st.plotly_chart( fig, use_container_width=True ) 

		html2 = '''<h2>✘ Errors Slope & Intercept</h2>'''
		st.markdown( html2, unsafe_allow_html=True )
		
		fig = go.Figure()
		fig.add_trace(go.Scatter(
			x=anew, y=(slope - A), connectgaps=True ) )
		fig.add_trace(go.Scatter(
			x=bnew, y=(intercept - B) ) )
		fig.update_layout( plot_bgcolor='#0e1117' )
		st.plotly_chart(fig, use_container_width=True)

		return None
		
	
if __name__ == '__main__':
	# ETL
	gradient = Grad()
	
	gradient.set_html()

	info = gradient.geral_gradient()

	x = info['x']
	y = info['y']
	A = info['A']
	B = info['B']

	errors = info['errors']
	slope  = info['slope']
	intercept = info['intercept']
	gradA= info['gradA']
	gradB= info['gradB']
	anew = info['anew']
	bnew = info['bnew']
	enew = info['enew']
	
	
	gradient.plot_charts( x, y, A, B, errors, slope, intercept, gradA, gradB, anew, bnew, enew )
