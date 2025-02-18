export default {
  linearGradient: {
    top: 0,
    left: 0,
    right: 0,
		height: 248,
		position: 'absolute',
  },
  imageBackdrop: {
    height: 248,
    backgroundColor: 'black'
  },
  cardContainer: {
    position: 'absolute',
		top: 32,
		right: 16,
		left: 16,
		flexDirection: 'row'
  },
  cardImage: {
		height: 184,
		width: 135,
		borderRadius: 3
	},
  cardDetails: {
		paddingLeft: 10,
		flex: 1
	},
  cardTitle: {
    color: 'white',
    fontSize: 19,
		fontWeight: '500',
		paddingTop: 0
  },
  cardGenre: {
		flexDirection: 'row'
	},
  cardGenreItem: {
		fontSize: 11,
		marginRight: 5,
		color: 'white'
	},
	cardDescription: {
		color: '#f7f7f7',
		fontSize: 13,
		marginTop: 5
	},
	cardNumbers: {
		flexDirection: 'row',
		marginTop: 5
	},
	cardStar: {
		flexDirection: 'row'
	},
	cardStarRatings: {
		marginLeft: 5,
		fontSize: 12,
		color: 'white'
	},
	cardRunningHours: {
		marginLeft: 5,
		fontSize: 12
	},
	viewButton: {
		justifyContent: 'center',
		padding: 10,
		borderRadius: 3,
		backgroundColor: '#EA0000',
		width: 100,
		height: 50,
		marginTop: 10
	},
	viewButtonText: {
		color: 'white'
	}
}
