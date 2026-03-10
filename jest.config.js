export default {
  testEnvironment: 'jsdom',
  transform: {
    '^.+\\.jsx?$': ['babel-jest', {
      presets: [
        ['@babel/preset-env', { targets: { node: 'current' } }],
        ['@babel/preset-react', { runtime: 'automatic' }],
      ],
    }],
  },
  moduleFileExtensions: ['js', 'jsx', 'json'],
  testMatch: ['**/tests/**/*.{test,spec}.{js,jsx}', '**/tests/**/test_*.{js,jsx}'],
  setupFilesAfterSetup: [],
};
