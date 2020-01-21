import React, { Component } from 'react';

import {Platform,
  StyleSheet,
  Text,
  View,
  TouchableOpacity
} from 'react-native';

import FBSDK, {
  LoginManager,
  LoginButton,
  AccessToken,
} from 'react-native-fbsdk';

class FacebookService {

  makeLoginButton(callback) {
    return (
      <LoginButton
      readPermissions={["public_profile", "user_photos", "user_posts", "user_events", "user_likes"]}
        onLoginFinished={
          async (error, result) => {
            if (error) {
            } else if (result.isCancelled) {
              console.log(result);
              alert("login is cancelled.");
            }
             else {
              AccessToken.getCurrentAccessToken()
                .then((data) => {
                  // console.log(data.accessToken.toString());
                  // this.setState({username:data.accessToken.toString()});
                  // this.onLogin();
                  callback(data.accessToken)
                })
                .catch(error => {
                  console.log(error)
                })
            }
        }
      } />
    );
  }

  makeLogoutButton(callback) {
    return (
      <LoginButton onLogoutFinished={() => {
        callback()
      }} />
    )
  }
}

export const facebookService = new FacebookService();
