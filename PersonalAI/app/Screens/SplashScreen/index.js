import React from 'react';
import { View, Text, Image } from 'react-native';
import Colors from './../../Constants/Colors';

import FBSDK, {
  LoginManager,
  LoginButton,
  AccessToken,
} from 'react-native-fbsdk';

export default class SplashScreen extends React.Component {

  constructor(props) {
    super(props);

    this.state = {
      accessToken: null
    }
  }

  async componentDidMount() {
    AccessToken.getCurrentAccessToken()
    .then((data) => {
      this.setState({
        accessToken: data.accessToken
      })
    })
    .catch(error => {
      console.log(error);
    });
    const data = await this.performTimeConsumingTask();
    if(data !== null && this.state.accessToken) {
      this.props.navigation.replace('MovieHome');
    } else {
      this.props.navigation.replace('LoginScreen');
    }
  }


  performTimeConsumingTask = async() => {
    return new Promise((resolve) =>
      setTimeout(
        () => { resolve('result') },
        2000
      )
    )
  }

  render() {
    return (
      <View style={styles.viewStyles}>
         <Image source={require('./React-Native.png')} style={styles.imageStyles}/>
        <Text style={styles.textStyles}>
          Movies
        </Text>
      </View>
    );
  }

}

const styles = {
  viewStyles: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.Black
  },
  imageStyles: {
    width: 200,
    height: 200
  },
  textStyles: {
    color: 'white',
    fontSize: 40,
    fontWeight: 'bold'
  }
}
